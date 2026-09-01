"""The tool-calling loop.

Written out rather than delegated to a framework: it is a dozen lines, and
keeping the stopping condition and the tool-result plumbing visible matters
more here than the abstraction would save. The safety property — that no tool
writes — is only obvious if you can see every tool the loop can reach.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm_client import LLMClient, LLMUsage, TextDelta, llm_client
from app.agent.prompts.assistant import ASSISTANT_SYSTEM_PROMPT
from app.agent.tools import TOOL_SCHEMAS, run_tool
from app.agent.tracing import hide, traced
from app.domain.enums import MessageRole
from app.models.conversation import AgentMessage

logger = logging.getLogger(__name__)

# Enough for a coherent thread without paying for the whole history on every
# turn. Older context is rarely load-bearing in this domain — "and Amazon?"
# refers to the last few messages, not to something said twenty turns ago.
HISTORY_TURNS = 10

# Turns alone are the wrong unit once a message can be 10,000 characters: ten of
# those replay ~25,000 tokens on every subsequent request, so one pasted posting
# is paid for again on each turn that follows it. The budget keeps the newest
# turns and drops the oldest, which is also the right order to lose them in.
HISTORY_CHARS = 12_000

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
    # Documents a tool pulled out of the database, shown to the user as they are
    # stored. They never pass through the model's output, which is the only way
    # to guarantee the user sees the posting rather than the model's impression
    # of it.
    attachments: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    total_tokens: int = 0
    cached_tokens: int = 0


# --- What a turn emits while it runs ---------------------------------------
#
# A turn can take six rounds and most of them are spent waiting on a tool. The
# events below are what makes that wait legible: prose as it is written, and
# the name of whatever is being looked up in between.


@dataclass(slots=True)
class Delta:
    """Part of the reply, as the model writes it."""

    text: str


@dataclass(slots=True)
class ToolStarted:
    """A tool is about to run.

    Emitted before the call rather than after, because the point is to account
    for the pause rather than to report it once it is over.
    """

    name: str


@dataclass(slots=True)
class Superseded:
    """Everything streamed this round belonged to a tool call, not an answer.

    Models sometimes narrate before reaching for a tool — "let me check that" —
    and that sentence is not part of the reply. It is never saved to the
    transcript and would not survive a reload, so it is withdrawn rather than
    left on screen as something the conversation no longer contains.
    """


@dataclass(slots=True)
class Completed:
    """The finished turn. Always the last event, unless the turn raised."""

    result: AssistantResult


TurnEvent = Delta | ToolStarted | Superseded | Completed


def _turn_run(outputs: Any) -> dict[str, Any]:
    """Put the answer in the trace, not the several hundred fragments of it."""
    value = outputs.get("output") if isinstance(outputs, dict) else outputs
    events = value if isinstance(value, list) else [value]
    done = next((e for e in reversed(events) if isinstance(e, Completed)), None)
    if done is None:
        return outputs if isinstance(outputs, dict) else {"output": outputs}

    return {
        "message": done.result.message,
        "proposal": done.result.proposal,
        "tools_used": done.result.tools_used,
        "total_tokens": done.result.total_tokens,
    }


async def load_history(session: AsyncSession, user_id: uuid.UUID) -> list[dict[str, str]]:
    """Recent turns, oldest first, within both a turn and a character budget."""
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

    kept: list[dict[str, str]] = []
    budget = HISTORY_CHARS
    for message in rows:  # newest first, so the budget is spent on recent context
        budget -= len(message.content)
        if budget < 0 and kept:
            break
        kept.append({"role": message.role, "content": message.content})

    return list(reversed(kept))


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
    """The whole turn at once, for callers that are not streaming it.

    Drains the generator below rather than reimplementing it. There is one loop
    on purpose: the property this design rests on — that no tool the loop can
    reach writes — is only checkable if there is one place to check it.
    """
    result: AssistantResult | None = None
    async for event in stream_assistant(session, user_id, message, client=client):
        if isinstance(event, Completed):
            result = event.result

    if result is None:  # pragma: no cover - the generator's last act is Completed
        raise RuntimeError("The assistant loop ended without producing a result.")
    return result


@traced(
    "assistant_turn",
    process_inputs=hide("session", "client"),
    process_outputs=_turn_run,
)
async def stream_assistant(
    session: AsyncSession,
    user_id: uuid.UUID,
    message: str,
    *,
    client: LLMClient | None = None,
) -> AsyncIterator[TurnEvent]:
    """Answer, using tools as needed, and possibly propose a change.

    Emits the reply as the model writes it and names each tool as it runs; the
    last event carries the finished turn. Writes to the transcript happen once,
    at the end, so a turn that fails part-way leaves nothing behind.
    """
    model = client or llm_client

    history = await load_history(session, user_id)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": message},
    ]

    proposal: dict[str, Any] | None = None
    attachments: list[dict[str, Any]] = []
    tools_used: list[str] = []
    total_tokens = 0
    cached_tokens = 0
    reply_text = ""

    for _round in range(MAX_ROUNDS):
        streamed = ""
        assistant_message: dict[str, Any] = {}
        usage: LLMUsage | None = None

        async for chunk in model.stream_chat(messages=messages, tools=TOOL_SCHEMAS):
            if isinstance(chunk, TextDelta):
                streamed += chunk.text
                yield Delta(chunk.text)
            else:
                assistant_message, usage = chunk.message, chunk.usage

        if usage is not None:
            total_tokens += usage.total_tokens
            cached_tokens += usage.cached_tokens
            # A prefix that stops being cached is a silent 2x on the prompt bill
            # — it does not fail, it just costs more — so it is logged rather
            # than left to be noticed on an invoice.
            logger.debug(
                "round: %d prompt tokens, %d from cache (%.0f%%)",
                usage.prompt_tokens,
                usage.cached_tokens,
                usage.cache_hit_rate * 100,
            )

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            reply_text = (assistant_message.get("content") or "").strip()
            break

        # Anything written this round was a preamble to the tool call, not the
        # answer — take it back before the tools are announced.
        if streamed:
            yield Superseded()

        # The assistant turn must go back verbatim, tool_calls included, or the
        # follow-up tool results have nothing to attach to.
        messages.append(assistant_message)

        for call in tool_calls:
            name = call["function"]["name"]
            yield ToolStarted(name)
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            result = await run_tool(name, arguments, session, user_id, message=message)
            tools_used.append(name)

            output = result.output
            if result.proposal is not None:
                if proposal is None:
                    proposal = result.proposal
                elif KEEP_FIRST_PROPOSAL:
                    # Told to the model rather than dropped in silence. Asked to
                    # delete two applications it prepares both, one card renders,
                    # and it reports both as done — so one of them quietly did
                    # not happen and the transcript says otherwise.
                    output = (
                        "NOT PREPARED — a change is already awaiting confirmation this turn, "
                        "and only one can be shown at a time. Tell them you have prepared "
                        f"the first one, and that you will do this one ({name}) next once "
                        "they have confirmed it. Do NOT describe this one as ready."
                    )
            # Asking twice in one turn is common when the model retries a query;
            # showing the same posting twice is not.
            if result.attachment is not None and result.attachment not in attachments:
                attachments.append(result.attachment)

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

    # `message` is sent again even though the client has been assembling it from
    # deltas all along. The two disagree whenever the loop hit its round limit
    # or the last round was withdrawn, and the transcript is the version that
    # was saved — so the client renders this one and any drift resolves itself.
    yield Completed(
        AssistantResult(
            message=reply_text,
            proposal=proposal,
            attachments=attachments,
            tools_used=tools_used,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )
    )
