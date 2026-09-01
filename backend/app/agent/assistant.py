"""The tool-calling loop.

Written out rather than delegated to a framework: it is a dozen lines, and
keeping the stopping condition and the tool-result plumbing visible matters
more here than the abstraction would save. The safety property — that no tool
writes — is only obvious if you can see every tool the loop can reach.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm_client import LLMClient, llm_client
from app.agent.prompts.assistant import ASSISTANT_SYSTEM_PROMPT
from app.agent.tools import TOOL_SCHEMAS, run_tool
from app.domain.enums import MessageRole
from app.models.conversation import AgentMessage

logger = logging.getLogger(__name__)

# Enough for a coherent thread without paying for the whole history on every
# turn. Older context is rarely load-bearing in this domain — "and Amazon?"
# refers to the last few messages, not to something said twenty turns ago.
HISTORY_TURNS = 10

# The loop must terminate. Each round is one model call plus its tools, and a
# model that has not answered after this many rounds is looping rather than
# working — better to say so than to keep spending the token budget.
#
# Six rather than four because the useful chains got longer: "how am I doing and
# what should I chase" is analytics, then attention, then a timeline, then the
# answer. Four rounds cut those off mid-thought, which reads to the user as the
# assistant giving up rather than as a limit being hit.
MAX_ROUNDS = 6

# A proposal is a question to the user, and two questions in one reply cannot
# both be answered — the confirm card only renders one. If the model proposes
# twice in a turn the first is kept, because it is the one it explained.
KEEP_FIRST_PROPOSAL = True


@dataclass
class AssistantResult:
    message: str
    proposal: dict[str, Any] | None = None
    tools_used: list[str] = field(default_factory=list)
    total_tokens: int = 0


async def load_history(session: AsyncSession, user_id: uuid.UUID) -> list[dict[str, str]]:
    """Recent turns, oldest first."""
    rows = list(
        (
            await session.execute(
                select(AgentMessage)
                .where(AgentMessage.user_id == user_id)
                .order_by(desc(AgentMessage.created_at))
                .limit(HISTORY_TURNS)
            )
        )
        .scalars()
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in reversed(rows)]


async def save_turn(
    session: AsyncSession, user_id: uuid.UUID, role: MessageRole, content: str
) -> None:
    session.add(AgentMessage(user_id=user_id, role=role.value, content=content))
    await session.flush()


async def run_assistant(
    session: AsyncSession,
    user_id: uuid.UUID,
    message: str,
    *,
    client: LLMClient | None = None,
) -> AssistantResult:
    """Answer, using tools as needed, and possibly propose a change."""
    model = client or llm_client

    history = await load_history(session, user_id)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": message},
    ]

    proposal: dict[str, Any] | None = None
    tools_used: list[str] = []
    total_tokens = 0
    reply_text = ""

    for _round in range(MAX_ROUNDS):
        assistant_message, usage = await model.chat(messages=messages, tools=TOOL_SCHEMAS)
        total_tokens += usage.total_tokens

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            reply_text = (assistant_message.get("content") or "").strip()
            break

        # The assistant turn must go back verbatim, tool_calls included, or the
        # follow-up tool results have nothing to attach to.
        messages.append(assistant_message)

        for call in tool_calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            output, maybe_proposal = await run_tool(name, arguments, session, user_id)
            tools_used.append(name)
            if maybe_proposal is not None and not (proposal and KEEP_FIRST_PROPOSAL):
                proposal = maybe_proposal

            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
    else:
        logger.warning("Assistant hit the round limit for user %s", user_id)
        reply_text = (
            "I could not work that out in a reasonable number of steps. "
            "Try asking about one application at a time."
        )

    if not reply_text:
        reply_text = "Done." if proposal else "I did not find anything to say about that."

    await save_turn(session, user_id, MessageRole.USER, message)
    await save_turn(session, user_id, MessageRole.ASSISTANT, reply_text)

    return AssistantResult(
        message=reply_text,
        proposal=proposal,
        tools_used=tools_used,
        total_tokens=total_tokens,
    )
