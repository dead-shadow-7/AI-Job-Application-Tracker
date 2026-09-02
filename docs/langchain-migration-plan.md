# Migrating the LLM layer to LangChain — implementation plan

## 1. Scope decision + recommendation

### The one-line recommendation

**Adopt LangChain as the *transport and model* layer. Do not adopt it as the *control-flow* layer.**

Replace the hand-rolled HTTP/SSE machinery inside `llm_client.py` with `ChatOpenAI`, keeping `LLMClient`'s public surface byte-identical. Keep the hand-written assistant loop, keep `TOOL_SCHEMAS`, keep the pinned `response_format` dict, keep `tracing.py`. Add `langchain-core` + `langchain-openai`; do **not** add `langchain`, `langchain-classic`, or `langchain-groq`.

### Does the whole of `llm_client.py` go?

**No. About 180 of its 540 lines go; the facade stays.**

| Goes | Stays | Why it stays |
|---|---|---|
| `_stream` :284-350 (SSE open + retry) | | commodity |
| `_read_stream` :352-434 (SSE framing, tool-call chunk reassembly) | | commodity; `AIMessageChunk.__add__` does it better |
| `_post` :436-498 HTTP mechanics | `_post`'s **retry policy** (413, `Retry-After`, `RuntimeError`) | `RETRYABLE_STATUS` :39 includes **413**; the openai SDK's retry set is `{408,409,429,≥500}` and **excludes 413**, which is precisely Groq's TPM-exhaustion code |
| | `_backoff` :519-537 | jitter-on-the-`Retry-After`-branch has no LangChain equivalent; `Runnable.with_retry()` defaults to `retry_if_exception_type=(Exception,)` and would retry 400s |
| | `_error_message` :500-517 | rewrites `code=="rate_limit_exceeded"` into user-facing prose; three tests match on the strings |
| | `get_http_client` :48 / `close_http_client` :73 | see below — these become an **asset**, not a liability |
| | `LLMError`, `LLMUsage`, `StructuredResult`, `TextDelta`, `TurnComplete`, `is_configured`, `extract`, `stream_chat` | ~70 tests and 4 route entry points bind to these names |

The decisive point on the pool: **pin `openai>=1.99,<3` and pass `http_async_client=get_http_client()` into `ChatOpenAI`.** That single decision resolves the httpx-vs-httpx2 blocker by *choosing* rather than discovering, and it carries four behaviours across for free — the 20/10 connection pool (:67), the split `Timeout(90, connect=10.0)` (:66), the event-loop keying (:61, the `RuntimeError: Event loop is closed` hazard), and the `close_http_client()` lifespan hook (`main.py:9,37`). It also keeps test Seam A alive unchanged.

Corollary: **construct `ChatOpenAI` per call, not once**, injecting `get_http_client()` each time. Caching the model object would freeze a loop-bound pool inside it and reintroduce exactly the bug `get_http_client`'s docstring :49-58 exists to prevent. Construction is cheap when the httpx client is supplied; the TCP/TLS handshake was the cost being avoided, and it still is.

### Does extraction keep the pinned-dict escape hatch?

**Yes. Emphatically. Do not use `with_structured_output` at all.**

Use `model.bind(response_format={"type": "json_schema", "json_schema": {"name": schema.__name__.lower(), "strict": True, "schema": to_strict_json_schema(schema)}})` — the exact dict currently at `llm_client.py:209-216`. The research confirms this is returned **byte-for-byte unchanged** by `_convert_to_openai_response_format` (base.py:4263-4268).

Three independent reasons, any one sufficient:

1. **`$ref` inlining has no LangChain equivalent.** `_inline_refs` (`extraction.py:160-181`) exists because Groq's validator cannot resolve refs. `ExtractedJob` nests `ExtractedSalary` (`extraction.py:104`) and `list[ExtractedRequirement]` (`:113`) — both arrive as `$ref` from `model_json_schema()`. LangChain's `_recursive_set_additional_properties_false` walks `$defs` but **does not inline**. Passing the Pydantic class means every real extraction 400s while CI stays green, because Seam C (`StubLLM`) bypasses `extract` entirely.
2. **The schema `name` changes.** The openai SDK's `.parse()` path forces `name = T.__name__` → `"ExtractedJob"`, where the repo sends `"extractedjob"` (`:212`). Cosmetic against OpenAI; unverified against the AI Credits gateway.
3. **Zero gain.** `with_structured_output(include_raw=True)` buys a parsed object and `raw.usage_metadata`. The repo already parses (`:224`) and already translates `ValidationError` → `LLMError` with a message the ingestion graph's retry edge reads (`ingestion.py:122-123`). Adopting it means re-deriving that translation from `parsing_error`.

`RubricJudgment` (`app/schemas/matching.py:7-22`) is flat and would survive `with_structured_output` — but there is no reason to run two different structured-output mechanisms in one codebase.

### Does the assistant loop become `create_agent`?

**No — and it should not become a hand-authored `StateGraph` either, in this migration.**

`create_agent` is out because it loses a **safety property**, not just conveniences:

> **All 22 tools share one `AsyncSession` carrying the transaction-local `app.user_id` GUC that every RLS policy reads** (`db/session.py:37-60`). The loop dispatches strictly serially (`assistant.py:260-291` — a plain `for` with `await` inside, no `gather`). LangGraph's `ToolNode` executes tool calls **concurrently by default**. Concurrent use of one SQLAlchemy `AsyncSession` raises `InterfaceError`/`IllegalStateChangeError`; and `db_pool_size` is 10 with `max_overflow` 4, so the "just give each tool its own session" fix is a pool-exhaustion and an RLS-scoping problem at once.

Mitigation *if* `create_agent` were forced: a `@wrap_tool_call` middleware holding an `asyncio.Lock` for the turn, plus `ToolRuntime[AgentContext]` on all 22 signatures to inject the session. That is more code than the current loop, and it expresses "serial" indirectly — a future middleware reorder silently reintroduces concurrency against RLS. Not worth it.

Five further behaviours `create_agent` has no native form for:

| Behaviour | Location | Why `create_agent` cannot |
|---|---|---|
| `Superseded` retraction | `assistant.py:252-254`, emitted **before** any `ToolStarted` (`test_agent_stream.py:131`) | No "withdraw the text I already streamed you" concept anywhere in LangGraph |
| One-proposal-per-turn arbitration | `:272-285` — the **second** call *in list order* has its `ToolMessage` content replaced with the `NOT PREPARED` instruction (`test_agent_actions.py:492-511`) | Requires ordered, stateful cross-tool arbitration inside one dispatch batch |
| `ToolStarted` before the call | `:262`, deliberately (docstring `:86-89`) | `ToolNode` emits after |
| Round-limit canned reply | `for...else` at `:292-297` | `ModelCallLimitMiddleware` stops the loop; it does not substitute *that* sentence (`test_agent.py:177-186`) |
| Character-budget history | `load_history` :131-154 — `HISTORY_TURNS=10` **and** `HISTORY_CHARS=12_000`, newest-first, "keep at least one" floor at `:150` | `trim_messages` counts tokens, has no floor, and operates on state — history here comes from an RLS-scoped DB query per user (`test_agent.py:82-91`) |

A hand-authored `StateGraph` for the assistant is *also* deferred, and this is the less obvious call, so the reason matters: the loop's topology is a two-node cycle whose only conditional edge is `if tool_calls`. There is no checkpointer, no interrupt, no store, no persistence requirement — LangGraph earns its place in `ingestion.py` because the retry edge and per-node state are genuinely non-trivial, and it would earn none here. Converting `stream_assistant` from an `AsyncIterator[TurnEvent]` generator to `astream(stream_mode=["messages","custom"])` would force all three ordering invariants (`Superseded` before `ToolStarted`, `ToolStarted` before dispatch, `Completed` last) through `get_stream_writer`, where they become implicit, and would invalidate ~70 tests in one commit. **Document it as a deliberate non-goal in the module docstring** rather than leaving the absence to look like an oversight.

### Do the tools get redefined with `@tool`?

**No.** Keep `TOOL_SCHEMAS`, `_tool`, `_nullable`, the three registries, and the import-time coverage assert (`tools.py:653-655`).

- `_nullable` (`:94-102`) emits `"type": ["string", "null"]` and appends `None` to enums. Pydantic emits `anyOf: [{"type":"string"},{"type":"null"}]` for `str | None` — a **different schema**, unverified against Groq's tool-argument validator, which per `tools.py:66-74` rejects the whole request with a 400 on disagreement. ~30 optional parameters across 22 tools.
- `run_tool` (`tools.py:382-419`) is **one** place where `session`, `user_id` and `message` are injected and where "no tool writes" is checkable. `ToolRuntime` would move that to 22 signatures. The inventory calls this "the migration's biggest single simplification and its biggest single risk" — it is the second, and the simplification is illusory because `run_tool` is 20 lines.
- Bind them with `.bind(tools=TOOL_SCHEMAS, tool_choice="auto")`, **not** `bind_tools()`, so `convert_to_openai_tool` never touches the dicts.

### Dependency set (final)

```
langchain-core >= 1.6.1
langchain-openai >= 1.6.0
langgraph >= 1.2.11        # was >=0.2.60 — forced by langchain-core>=1.4.7,<2
langsmith >= 0.3.45        # was >=0.2
openai >= 1.99, <3         # pinned: openai 3.x is httpx2; see Phase 1
httpx >= 0.27              # unchanged
```
No `langchain`. No `langchain-classic`. No `langchain-groq` — `ChatGroq` defaults to `method="function_calling"` and silently downgrades `strict=True` to `None` for every model except `openai/gpt-oss-*` (chat_models.py:904, 1214), which is *literally the failure mode* `llm_client.py:1-9` was written to avoid. One `ChatOpenAI` class, three base URLs, matching `config.py:150-170`.

---

## 2. Phased sequence

Every phase ships alone and leaves the suite green. Ordering principle: **the untested contract gets a test before anything moves**, then the seam-preserving infrastructure, then the two call paths independently (so either can be reverted without the other), then deletion.

---

### Phase 0 — Test the strict-schema contract *before* touching anything

**Goal.** Turn the three untested, load-bearing behaviours into golden tests that the LangChain adapter must reproduce. Without this, Phases 3-4 are unverifiable: the inventory §7 records that `to_strict_json_schema` has **no direct test**, and Seam C stubs `extract` outright, so `llm_client.py:209-216` is **never exercised in CI**.

**Files touched (all new).**
- `backend/tests/test_extraction_schema.py`
- `backend/tests/test_llm_payload.py`
- `backend/tests/test_llm_retry.py`

**The change.**

`test_extraction_schema.py` — direct assertions on `to_strict_json_schema(ExtractedJob)` and `(RubricJudgment)`:
- `"$defs" not in schema`, and no `"$ref"` anywhere in the tree (recursive walk).
- Every node with `type == "object"` and `properties` has `additionalProperties is False` and `required == list(properties)` — checked at **every** depth, including inside `salary` and `requirements.items`.
- `schema["properties"]["salary"]["properties"]["raw_text"]` exists (proves `ExtractedSalary` was inlined, not left as a ref).
- `schema["properties"]["requirements"]["items"]["properties"]["kind"]["enum"] == ["must","nice"]`.
- Sibling-key merge: a `$ref` node carrying a `description` keeps it (`extraction.py:174`).
- Depth guard: a self-referencing model raises `ValueError` (`:167-168`).

`test_llm_payload.py` — capture the outgoing `/chat/completions` JSON via `httpx.MockTransport` and assert:
- Extraction: `payload["response_format"] == {...exact dict...}` including `"name": "extractedjob"` and `"strict": True`; `payload["max_completion_tokens"] == 3000`; `payload["temperature"] == 0.0`; `"stream" not in payload`.
- Chat: `payload["stream"] is True`; `payload["stream_options"] == {"include_usage": True}`; `payload["tools"] == TOOL_SCHEMAS`; `payload["tool_choice"] == "auto"`; `payload["max_completion_tokens"] == 1024`.
- Headers: `Authorization: Bearer <key>`; URL is `{base_url}/chat/completions`.
- One assertion that `TOOL_SCHEMAS` contains at least one `"type": ["string","null"]` and at least one enum with `None` appended — pinning `_nullable`.

`test_llm_retry.py` — the untested retry semantics:
- 413 with body `{"error":{"code":"rate_limit_exceeded","message":"..."}}` retries, and `_error_message` returns `"Groq rate limit reached. …"`.
- `Retry-After: 5` is honoured: monkeypatch `asyncio.sleep` and `random.uniform`→`lambda a,b: a`, assert slept `5.0`; `Retry-After: abc` falls back to `min(2**attempt, 8)`.
- Three failures raise `LLMError` matching `"unreachable after 3 attempts"`, and **no sleep occurs after the last attempt** (assert exactly 2 sleeps).
- A bare `RuntimeError` from the transport is retried (`:450`).

**Verification.** `cd backend && python -m pytest tests/test_extraction_schema.py tests/test_llm_payload.py tests/test_llm_retry.py -v` then the full suite.

**Rollback.** Delete three files. Zero production risk — nothing in `app/` changes.

**Size.** 3 files, +260 LOC, 0 production LOC.

---

### Phase 1 — Dependency bump and LangGraph 1.x compatibility

**Goal.** Land the version wall by itself, so a LangGraph 1.x regression in the ingestion DAG is a one-line revert and not entangled with a client rewrite.

**Files touched.**
- `backend/pyproject.toml:20-29`
- `backend/app/agent/graphs/ingestion.py:200,210,232`
- `backend/tests/conftest.py:52-60`

**The change.**

`pyproject.toml` — replace the "deliberately no langchain-groq" comment block (`:23-27`) with an updated rationale, and set the dependency list above. Keep the anti-`langchain-groq` note: it is now *more* justified, not less (chat_models.py:904, 1214).

```toml
"langgraph>=1.2.11",
"langsmith>=0.3.45",
"langchain-core>=1.6.1",
"langchain-openai>=1.6.0",
# openai 3.x is built on the `httpx2` distribution, while langchain-core still
# depends on httpx<1.0 — both install side by side. Pinning <3 keeps the pooled
# `httpx.AsyncClient` in agent/llm_client.py injectable as `http_async_client`,
# which is what carries the connection pool, the split timeout and the
# event-loop keying across from the hand-rolled client.
"openai>=1.99,<3",
```

`ingestion.py`:
- `:19` → `from langgraph.graph import END, START, StateGraph`
- `:210` `graph.set_entry_point("normalize")` → `graph.add_edge(START, "normalize")`
- `:200` and `:232` return types `-> Any` → `-> CompiledStateGraph[IngestionState]` (fall back to `Any` if the generic fights mypy under `disallow_untyped_defs`).
- Everything else — `IngestionState`, the `Annotated[list[LLMUsage], lambda a, b: a + b]` reducer (`:58`), the `session: Any`/`client: Any` state fields (`:47-48`), both `add_conditional_edges`, the module-global compiled cache (`:229-237`), the `config={"run_name","metadata"}` at `:260-267` — is unchanged. LangGraph v1 is "largely backwards compatible"; `StateGraph` is untouched; the repo uses no `create_react_agent`/`AgentState`/`MessageGraph`/`ValidationNode`.

`conftest.py:52-60` — extend the tracing kill-switch. LangChain reads more env than `LANGSMITH_TRACING`/`LANGCHAIN_TRACING_V2`:
```python
for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING",
            "LANGSMITH_OTEL_ENABLED"):
    os.environ[var] = "false"
for var in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_PROJECT",
            "LANGCHAIN_PROJECT", "LANGSMITH_ENDPOINT"):
    os.environ.pop(var, None)
```
Still **before any `app.*` import** — the SDK reads env at import.

**New/changed tests.**
- `test_llm_providers.py:92-103` (`test_the_suite_does_not_write_traces`) extended to assert all four flags false and all five keys absent.
- New `test_ingest.py::test_the_graph_still_starts_at_normalize` — `get_ingestion_graph().get_graph()` contains an edge from `START` to `"normalize"`.

**Verification.**
```
cd backend && pip install -e ".[dev]" && pip list | grep -E "langgraph|langchain|langsmith|openai|httpx"
python -m pytest tests/ -q
python -c "import langchain_openai, httpx; from langchain_openai.chat_models import base; assert base.httpx is httpx"
```
That last line is the httpx2 tripwire — add it as a test (`test_dependencies.py::test_langchain_and_openai_share_one_httpx`) so the day the `<3` pin is lifted, CI says so instead of Seam A silently breaking.

**Rollback.** `git revert`; the graph delta is two lines.

**Size.** 3 files, +25 / −8 LOC.

---

### Phase 2 — Chat-model factory, not yet wired to production

**Goal.** Get the riskiest configuration decisions (`stream_usage`, `max_retries`, `max_completion_tokens`, loop-keyed pooling) under test before any call path depends on them.

**Files touched.**
- `backend/app/agent/models.py` (new)
- `backend/tests/test_chat_model.py` (new)

**The change.** One function:

```python
def build_chat_model(*, model: str, max_output_tokens: int,
                     temperature: float = 0.0, streaming: bool = False) -> Runnable:
```

Behaviour, each line load-bearing:

| Setting | Value | Why |
|---|---|---|
| `base_url` | `settings.llm_base_url` | `config.py:177` — unchanged provider table |
| `api_key` | `settings.llm_api_key` | `config.py:173` |
| `http_async_client` | `get_http_client()` — **fetched at call time, never cached** | preserves pool (`:67`), split timeout (`:66`) and loop keying (`:61`) |
| `stream_usage` | `True`, **explicitly** | 🚨 auto-disabled whenever `openai_api_base` is set *or* a custom http client is injected (base.py:1332) — **both are true here**. Without it every streamed turn reports zero tokens and LangSmith prices at zero, exactly the silent triple failure `llm_client.py:270-273` documents |
| `max_retries` | `0` | we keep `_backoff` + `RETRYABLE_STATUS`; the SDK's set excludes **413** |
| `timeout` | `httpx.Timeout(settings.llm_timeout_seconds, connect=10.0)` | belt-and-braces; the SDK sends a per-request timeout that overrides the client default |
| `max_tokens` | **`None`** | do **not** use it |
| `.bind(max_completion_tokens=N)` | always | ⚠️ `ChatOpenAI.max_tokens` maps to the deprecated `max_tokens` wire field. Groq debits **`max_completion_tokens`** against TPM *before* generation (`config.py:191-193`), so the field name is a cost knob, not a synonym. Phase 0's golden payload test catches this |
| `streaming` | caller's choice | |

Return `model.bind(max_completion_tokens=max_output_tokens)` so the caller gets a `Runnable` with the ceiling already fixed.

**New tests** (`test_chat_model.py`):
- Each of the three providers yields the right `openai_api_base` / key / model — reusing `Settings(_env_file=None, LLM_PROVIDER=...)` as `test_llm_providers.py:16-29` does.
- `model.stream_usage is True` (regression-locks the base.py:1332 trap).
- `model.max_retries == 0`.
- `model.max_tokens is None` and the bound kwargs contain `max_completion_tokens`.
- Calling `build_chat_model` twice on the same loop reuses one `httpx.AsyncClient` object; calling it on a fresh loop builds a new one.
- Construction does **not** raise when `settings.llm_api_key == ""` (the `is_configured` gate at `api/v1/agent.py:85,133`, `ingest.py:46`, `matching.py:116` must stay the thing that produces the friendly message, not a constructor `ValueError`). If `ChatOpenAI` insists on a non-empty key, pass `api_key=settings.llm_api_key or "not-configured"` and let the guard above it do its job.

**Verification.** `python -m pytest tests/test_chat_model.py tests/test_llm_providers.py -v`, then full suite. Production untouched — `models.py` has no importers yet.

**Rollback.** Delete two files.

**Size.** 2 files, +90 / +110 test LOC.

---

### Phase 3 — Extraction path onto `ChatOpenAI`

**Goal.** Move `LLMClient.extract` (`llm_client.py:177-229`) onto `ChatOpenAI` while keeping its signature, its `StructuredResult[T]` return, its `LLMError` messages, and its exact wire payload. `stream_chat` is untouched — the two paths move separately so either can be reverted alone.

**Files touched.** `backend/app/agent/llm_client.py` only.

**The change.**

```python
model = build_chat_model(model=target_model,
                         max_output_tokens=max_tokens or settings.llm_max_output_tokens,
                         temperature=temperature)
bound = model.bind(response_format={          # ← the dict from :209-216, verbatim
    "type": "json_schema",
    "json_schema": {"name": schema.__name__.lower(), "strict": True,
                    "schema": to_strict_json_schema(schema)},
})
message = await self._invoke_with_retry(bound, [SystemMessage(system), HumanMessage(user)],
                                        target_model)
```

`_invoke_with_retry` is `_post`'s loop (`:436-498`) with the httpx call swapped for `await bound.ainvoke(...)` and the status branch swapped for exception mapping:

```python
except openai.APIStatusError as exc:
    message = self._error_message(exc.response)          # same reader, same strings
    if exc.status_code in RETRYABLE_STATUS: ...          # 413 stays retryable
        await self._backoff(attempt, exc.response.headers.get("retry-after"))
except (openai.APIConnectionError, httpx.RequestError, RuntimeError) as exc: ...
```
`_error_message` (`:500-517`) takes an `httpx.Response` today and `exc.response` is one — no change to that function at all. This is why the `openai<3` pin matters: under httpx2 it would be a different class.

Then, in order:
1. **`finish_reason` guard, reimplemented explicitly** — `if message.response_metadata.get("finish_reason") == "length": raise LLMError(f"{model} hit the output ceiling before finishing. Ask for less at once, or raise LLM_CHAT_OUTPUT_TOKENS.")`. LangChain returns a normal `AIMessage`; the string must match `:464-467` character for character.
2. `LLMUsage` from `message.usage_metadata`: `input_tokens`/`output_tokens`/`total_tokens`, and `cached_tokens = (usage_metadata.get("input_token_details") or {}).get("cache_read", 0)`.
3. `latency_ms` from `perf_counter()` deltas. `response.elapsed` (`:475`) is gone. **Deliberate small change**: the number now includes client-side serialisation (~1ms) and matches how the streamed path already measures (`:296,432`).
4. `schema.model_validate_json(message.text)` inside the existing `try/except ValidationError` → `LLMError` (`:223-229`), unchanged.
5. `record_model(usage.model, settings.llm_provider)` at `:220`, unchanged.

**Changed tests.**
- `test_llm_payload.py` (Phase 0) is **re-pointed**, not rewritten: `install()` swaps `monkeypatch.setattr("app.agent.llm_client.get_http_client", ...)` for `monkeypatch.setattr("app.agent.models.get_http_client", ...)`. Same `MockTransport`, same assertions. This is the whole payoff of Phase 0 — the golden payload test proves the LangChain path emits the same bytes.
- `test_llm_retry.py` grows a case: an `openai.APIStatusError` carrying a 413 body maps to the same message.
- **New**: `test_extract_raises_on_truncation` — mock a 200 with `finish_reason: "length"`, assert `LLMError` matching `"output ceiling"`. This gap exists today for the non-streamed path (only the streamed half is covered at `test_llm_client.py:65-96`), and Phase 3 is where it would silently vanish.
- Seam C (`StubLLM`, 10 patch sites in `test_ingest.py` + `test_agent_actions.py:886-965`) — **untouched**, because `extract`'s signature and return type do not move.

**Verification.**
```
python -m pytest tests/test_llm_payload.py tests/test_llm_retry.py tests/test_ingest.py tests/test_matching.py -v
python -m pytest tests/ -q
```
Plus a **live probe** — the one thing CI cannot do (see §5, unknowns a and d):
```
LLM_PROVIDER=groq      python -m scripts.probe_extract   # ad-hoc, not committed
LLM_PROVIDER=gemini    python -m scripts.probe_extract
LLM_PROVIDER=aicredits python -m scripts.probe_extract
```
one real posting each, asserting a valid `ExtractedJob` and non-zero `usage.cached_tokens` on the second run.

**Rollback.** `git revert` — one file, and `stream_chat` never moved, so ingestion and matching fall back to `_post` intact. **This is why Phase 5 must not delete `_post` until Phase 3 has run live against all three providers.**

**Size.** 1 file, +60 / −45 LOC. Tests +50.

---

### Phase 4 — Chat/streaming path onto `ChatOpenAI`

**Goal.** Replace `_stream` (`:284-350`) and `_read_stream` (`:352-434`) with `bound.astream(...)`, while `stream_chat` continues to yield `TextDelta` and exactly one `TurnComplete`. **`assistant.py` does not change. `ScriptedLLM` does not change. ~70 tests do not change.**

**Files touched.** `backend/app/agent/llm_client.py` only.

**The change.**

```python
bound = build_chat_model(model=..., max_output_tokens=settings.llm_chat_output_tokens,
                         streaming=True)
if tools:
    bound = bound.bind(tools=tools, tool_choice="auto")   # raw dicts, NOT bind_tools
```
Then, inside `_stream`'s existing retry/`delivered` skeleton (which is kept verbatim — the "half an answer is on screen" rule at `:337-344` is not something LangChain has an opinion about):

```python
full: AIMessageChunk | None = None
async for chunk in bound.astream(lc_messages):
    full = chunk if full is None else full + chunk        # accumulates text, tool_call_chunks, usage
    if chunk.text and not chunk.tool_call_chunks:
        delivered = True
        yield TextDelta(chunk.text)
```
The filter is `chunk.text` **vs** `chunk.tool_call_chunks`, not node identity — the research is explicit that tool-call chunks arrive on the same channel.

Then rebuild the raw OpenAI-shaped dict `TurnComplete` currently produces at `:417-419`. Four details, each a silent bug if missed:

1. **`content` must be `None`, not `""`,** when the turn is tool-calls-only (`:417`). `AIMessageChunk.content` is `""`. `assistant.py:248` does `(assistant_message.get("content") or "").strip()` so it tolerates both, but `conftest.py:315`'s `says(...)`/`calls(...)` shapes and `test_llm_client.py:120-149` pin `None`. Map explicitly.
2. **`arguments` must be the raw accumulated string**, taken from `full.tool_call_chunks[i]["args"]` — **not** `json.dumps(full.tool_calls[i]["args"])`. `assistant.py:264-266` does `json.loads(...)` with a `JSONDecodeError → {}` fallback; re-serialising a parsed dict makes that branch unreachable and changes what a mid-JSON truncation does. Also map `full.invalid_tool_calls` (LangChain's bucket for unparseable args) into the same list, or the malformed-arguments path disappears.
3. **`id`** from the chunk, `type: "function"`, ordered by chunk index (`:419`).
4. **`finish_reason == "length"`** read from `full.response_metadata` → the same `LLMError` string as `:411-415`, raised **after** the stream drains and **before** `TurnComplete` is yielded.

Usage: `full.usage_metadata` arrives as a trailing chunk (that is what `stream_usage=True` buys). `cached_tokens` from `input_token_details["cache_read"]`. `latency_ms` from `perf_counter()` — unchanged (`:296,432`).

**Changed tests.**
- `test_llm_client.py` (170 lines, 5 tests) — `install()` re-points to `app.agent.models.get_http_client` as in Phase 3; **the SSE response bodies stay identical** because they are wire-level, and **all five assertions stay identical**. The two truncation tests (`:65`, `:74`), the fragment-reassembly test (`:99`), the split-arguments test (`:120`) and the streamed-400-message test (`:152`) are the acceptance criteria for this phase. The last one is the sharpest: it currently asserts that `await response.aread()` before `_error_message` preserves the provider's own explanation (`:310-313`); post-migration it asserts the same through `openai.APIStatusError.response`.
- Seam B (`ScriptedLLM`, `conftest.py:293-326`, and the `llm` fixture's two patch targets at `:332-333`) — **untouched**.
- `test_agent.py` (16 tests), `test_agent_actions.py` (~50), `test_agent_stream.py` (8) — **untouched**.

**Verification.** `python -m pytest tests/test_llm_client.py -v` then the full suite. Then a manual `/chat/stream` against a real provider watching for: word-level deltas arriving, `superseded` firing before `tool`, non-zero `total_tokens` in the `done` frame, and non-zero `cached_tokens` on a second turn.

**Rollback.** `git revert` — one file; extraction stays on the Phase 3 path.

**Size.** 1 file, +85 / −110 LOC. Tests: ~15 changed lines in one file.

---

### Phase 5 — Delete the hand-rolled wire layer

**Goal.** Remove what nothing calls. Only after Phases 3 and 4 have both run live against all three providers.

**Files touched.** `backend/app/agent/llm_client.py`, and its module docstring `:1-9`.

**Delete.** `_post` :436-498, `_stream`'s httpx internals, `_read_stream` :352-434, the SSE constants. ~180 LOC.

**Keep.** `get_http_client` :48, `close_http_client` :73 (still the pool `ChatOpenAI` is handed, still wired to `main.py:9,37`), `RETRYABLE_STATUS` :39, `_backoff` :519, `_error_message` :500, `LLMError`, `LLMUsage`, `StructuredResult`, `TextDelta`, `TurnComplete`, `LLMClient`, `llm_client` :540.

**Rewrite the docstring** `:1-9`. Its claim — *"a wrapper's `with_structured_output` may emit function-calling or loose JSON mode depending on version"* — was accurate for langchain-openai <0.3 and is now false for `ChatOpenAI` (which defaults `method="json_schema"`), **but remains true for `ChatGroq`** and for `BaseChatOpenAI`. The replacement should say: the pinned dict is kept because `_inline_refs` has no LangChain equivalent, and `_convert_to_openai_response_format` passes a pre-shaped dict through byte-for-byte, so pinning costs nothing.

**Verification.** Full suite; `ruff check`; `mypy app/`.

**Size.** 1 file, −180 LOC.

---

### Phase 6 — Tracing reconciliation

**Goal.** Keep the four `@traced` spans as the dashboard's source of truth while `ChatOpenAI` adds a nested run underneath, and make sure the nested run prices correctly.

**Files touched.** `backend/app/agent/llm_client.py`, `backend/app/agent/tracing.py`.

**The change.**

1. **`ls_provider` override.** Pass on every model call:
   ```python
   config={"run_name": "chat_completion",
           "metadata": {"ls_provider": settings.llm_provider,
                        "ls_model_name": target_model}}
   ```
   Without this the nested run reports `ls_provider="openai"` where every historical run says `"groq"`/`"gemini"`/`"aicredits"` — breaking cost attribution and every saved LangSmith filter. Keep `record_model` (`:220`, `:280`) for the outer span; the config handles the inner one.
2. **Keep `@traced("extract"|"chat"|"tool"|"assistant_turn")` and both output reshapers** (`_stream_run` :133, `_extract_run` :153, `_turn_run` `assistant.py:115`). The nested `ChatOpenAI` run is additive and genuinely useful — it carries `finish_reason` and `invocation_params`, which the hand-rolled spans never had.
3. **`configure_tracing` unchanged** (`tracing.py:26-105`). Nothing in LangChain replicates the boot-time credential check or the region-mismatch guidance (`:53-61`).
4. **The tools-payload upload.** `hide("self","tools")` (`:234`) protects the *outer* span; the nested run uploads `TOOL_SCHEMAS` into `invocation_params` on every round — ~13 KB × 6 rounds × every turn. Measure first. If it is a problem, the only tier that covers auto-generated runs is `Client(hide_inputs=callable, hide_outputs=callable)` — construct it in `configure_tracing` with a callable that strips `tools`/`invocation_params.tools`, and register it as the default client.

**New tests.** Tracing is off in CI, so test the *shape*, not the upload:
- `test_tracing.py::test_the_chat_config_names_the_real_provider` — assert the `config` dict `stream_chat` builds carries `ls_provider == settings.llm_provider` for each of the three, and never `"openai"`.
- `test_tracing.py::test_usage_metadata_carries_cache_read` — feed a canned `AIMessage` with `input_token_details={"cache_read": 900}` and assert `LLMUsage.cached_tokens == 900` and `cache_hit_rate` is right. This is the first test `cache_read` has ever had, and it is risk #6.

**Size.** 2 files, +30 / −5 LOC. Tests +45.

---

### Phase 7 — Cleanup and documentation

- `llm_client.py` module docstring finalised; `assistant.py:1-7` docstring gains an explicit **non-goal** paragraph: why the loop is not `create_agent` and not a `StateGraph` (serial dispatch under one RLS session; `Superseded`; proposal arbitration). Otherwise the next reader migrates it.
- `tools.py:1-22` docstring gains one line on why `@tool` was not adopted (`_nullable` vs Pydantic `anyOf`).
- `pyproject.toml` comment block finalised.
- README / architecture docs updated if they name `groq_client.py` (the `pyproject.toml:25` comment already references a stale path).

**Size.** 4 files, docs only.

---

## 3. Behaviour-preservation checklist

| # | Behaviour | Current location | Preserved how | Test that proves it |
|---|---|---|---|---|
| 1 | Exact `response_format` incl. `name="extractedjob"`, `strict:true` | `llm_client.py:209-216` | `.bind(response_format=<pinned dict>)`; passed through byte-for-byte by `_convert_to_openai_response_format` | **new** `test_llm_payload.py` (Phase 0) |
| 2 | `$ref` inlining + `$defs` drop | `extraction.py:154-181` | `to_strict_json_schema` kept and called; LangChain never sees the Pydantic class | **new** `test_extraction_schema.py` |
| 3 | `additionalProperties:false` + full `required` at every depth | `extraction.py:184-194` | same | **new** `test_extraction_schema.py` |
| 4 | 413 retried as TPM exhaustion | `llm_client.py:39, 480, 513-514` | `RETRYABLE_STATUS` + `_error_message` kept; SDK retries disabled (`max_retries=0`) | **new** `test_llm_retry.py` |
| 5 | `Retry-After` honoured, jitter only adds | `:519-537` | `_backoff` kept verbatim; header read from `exc.response.headers` | **new** `test_llm_retry.py` |
| 6 | Bare `RuntimeError` retried ("event loop is closed") | `:450` | kept in `_invoke_with_retry`'s except clause | **new** `test_llm_retry.py` |
| 7 | No sleep after the final attempt | `:438, 453, 491` | `final_attempt` guard kept | **new** `test_llm_retry.py` |
| 8 | Stream retries only while undelivered | `:297, 334, 337-344` | `delivered` flag wraps `astream` unchanged | `test_llm_client.py` (extend) |
| 9 | Provider's own message survives a streamed non-200 | `:310-313` | `openai.APIStatusError.response` → same `_error_message` | `test_llm_client.py:152-169` |
| 10 | Pooled, loop-keyed httpx client | `:48-70` | **kept**; injected as `http_async_client`; `ChatOpenAI` built per call, never cached | `test_chat_model.py` (Phase 2) |
| 11 | Split timeout 90s read / 10s connect | `:66` | same client; also mirrored on `ChatOpenAI(timeout=)` | `test_chat_model.py` |
| 12 | `close_http_client()` in the lifespan | `:73`, `main.py:9,37` | **kept**, unchanged | existing lifespan tests |
| 13 | `finish_reason=="length"` raises, non-streamed | `:463-467` | explicit check on `response_metadata` | **new** `test_extract_raises_on_truncation` |
| 14 | `finish_reason=="length"` raises, streamed | `:411-415` | explicit check on accumulated `full.response_metadata` | `test_llm_client.py:65-96` |
| 15 | Streamed tool-call reassembly, `arguments` concatenated | `:393-405` | `AIMessageChunk.__add__`; raw string from `tool_call_chunks[i]["args"]` | `test_llm_client.py:120-149` |
| 16 | `content is None` when tool-calls-only | `:417` | explicit `or None` mapping | `test_llm_client.py:120-149` |
| 17 | `stream_options.include_usage` | `:274` | 🚨 `stream_usage=True` **explicit** (auto-off with `openai_api_base` + custom http client, base.py:1332) | `test_chat_model.py`; `test_llm_payload.py` |
| 18 | `cached_tokens` from `prompt_tokens_details` | `:421, :474` | `usage_metadata["input_token_details"]["cache_read"]`, `.get(…, 0)` | **new** `test_tracing.py::test_usage_metadata_carries_cache_read` |
| 19 | `ValidationError` → `LLMError`, no repair | `:223-229` | we parse ourselves; `with_structured_output` never used | `test_ingest.py` (existing failure paths) |
| 20 | Retry-on-`None` lives in the graph, capped at 2 | `ingestion.py:142-165` | untouched | `test_ingest.py:231, 259` |
| 21 | `is_configured` gate at 4 routes | `:167`; `agent.py:85,133`, `ingest.py:46`, `matching.py:116` | property kept on the facade; `ChatOpenAI` built lazily *after* the guard | `test_chat_model.py` + existing route tests |
| 22 | `LLMError` is the only exception escaping the client | `:81` | `openai.*` mapped inside `_invoke_with_retry`/`_stream`; **nothing** `openai`-typed leaves `llm_client.py` | **new** `test_llm_retry.py::test_no_openai_exception_escapes` |
| 23 | `matching.py` degrades to the 85% score | `matching.py:130-132` | ditto — it catches `LLMError` only | `test_matching.py` |
| 24 | `TOOL_SCHEMAS` sent verbatim, `tool_choice:"auto"` | `tools.py:140-358`; `:276-278` | `.bind(tools=…)`, **not** `bind_tools()` | `test_llm_payload.py` |
| 25 | `_nullable` widening `["string","null"]` + enum `None` | `tools.py:94-102` | `_tool`/`_nullable` untouched; no `@tool` | `test_llm_payload.py` |
| 26 | Import-time coverage assert | `tools.py:653-655` | untouched | runs on every import |
| 27 | Serial tool dispatch under one RLS session | `assistant.py:260-291` | **loop not migrated**; no `ToolNode` | `test_agent_actions.py:492-511`; `test_rls.py`; `test_tenancy.py` |
| 28 | `Superseded` before any `ToolStarted` | `assistant.py:252-254` | loop not migrated | `test_agent_stream.py:114-132` |
| 29 | One-proposal-per-turn, `NOT PREPARED` on the second | `assistant.py:272-285` | loop not migrated | `test_agent_actions.py:492-511` |
| 30 | Round limit 6 + canned reply | `assistant.py:221, 292-297` | loop not migrated | `test_agent.py:177-186` |
| 31 | History: 10 turns **and** 12 000 chars, keep ≥1 | `assistant.py:131-154` | loop not migrated; no `trim_messages` | `test_agent.py:65-91` |
| 32 | Transcript written once, at the end | `assistant.py:302-303` | loop not migrated | `test_agent_stream.py:147-158` |
| 33 | Unknown tool returns a string | `tools.py:414` | `run_tool` untouched | `test_agent.py:165-174` |
| 34 | `_clip` at 6 000 chars, announced | `tools.py:422-425` | untouched | `test_agent_actions.py` |
| 35 | SSE wire contract `start/delta/tool/superseded/done/error` | `agent.py:153-213`; `frontend/src/lib/api.js:79-139` | `TurnEvent` unchanged, so `_as_event` unchanged | `test_agent_stream.py` (8 tests) |
| 36 | `ls_provider` = the real provider | `:220,280`; `tracing.py:124-140` | `record_model` kept **+** per-invocation `config.metadata` override on the nested run | **new** `test_tracing.py` |
| 37 | Provider table drives everything | `config.py:142-186` | one `ChatOpenAI`, three base URLs | `test_llm_providers.py` (unchanged) |
| 38 | Suite writes no traces | `conftest.py:52-60` | extended to 4 flags + 5 keys | `test_llm_providers.py:92-103` (extended) |
| 39 | Whole-request deadlines | `agent.py:94,187`; `ingest.py:53` | untouched | — (untested today; see risk register) |
| 40 | Ingestion DAG topology + `usage` reducer | `ingestion.py:41-226` | `set_entry_point`→`add_edge(START,…)` only | `test_ingest.py` + **new** topology test |

### Deliberately dropped

| Dropped | Consequence | Why acceptable |
|---|---|---|
| `response.elapsed` for non-streamed `latency_ms` (`:475`) | `latency_ms` now includes client-side serialisation, ~1 ms | matches how the streamed path already measures (`:296,432`); nothing asserts on it |
| Hand-parsed SSE framing (`:368-379`), incl. "log and skip an unparseable chunk" | An unparseable chunk now raises from inside the openai SDK rather than being skipped | the SDK's parser is more tolerant than the hand-rolled one in every other respect; the "skip" branch has no test and no recorded incident |
| `_read_stream`'s `[DONE]` sentinel handling (`:372-373`) | — | SDK-internal now |
| `hide("self","tools")` coverage of the **nested** run | ~13 KB of tool JSON uploaded per round | measured in Phase 6; `Client(hide_inputs=…)` is the fix if it bites |
| Nothing else. | | |

---

## 4. Test strategy

### The four existing seams, and what happens to each

| Seam | What it is | Post-migration | When it moves |
|---|---|---|---|
| **A** — httpx `MockTransport` patching the accessor (`test_llm_client.py:29-45`, esp. `:44`) | wire-protocol truth, 5 tests | **Survives, re-pointed one line**: `monkeypatch.setattr("app.agent.llm_client.get_http_client", …)` → `"app.agent.models.get_http_client"`. The transport, the SSE bodies and every assertion are unchanged, because `http_async_client=get_http_client()` puts the mock underneath `ChatOpenAI` | Phase 3 (extract), Phase 4 (chat) |
| **B** — `ScriptedLLM` at `conftest.py:293-326`, patched at two globals (`:332-333`) | ~70 tests across `test_agent.py`, `test_agent_actions.py`, `test_agent_stream.py` | **Never moves.** `stream_chat`/`TextDelta`/`TurnComplete`/`is_configured` are preserved exactly. This is the single strongest argument for the adapter design | never |
| **C** — `StubLLM` at `test_ingest.py:70-100`, 10 patch pairs | ingestion + `propose_tracked_posting` | **Never moves.** `extract`'s signature and `StructuredResult[T]` are preserved | never |
| **D** — provider table, `test_llm_providers.py` | 6 config tests | **Survives intact**, including the `Literal`-exhaustiveness test at `:59-69` and the key/host-pairing test at `:44-56` — precisely because we use **one** `ChatOpenAI` with three base URLs rather than three provider classes | never (gains `test_chat_model.py` alongside) |

### The new seam: httpx `MockTransport`, not `FakeChatModel`

**Recommendation: httpx `MockTransport` as the sole new-code seam. Do not adopt `GenericFakeChatModel`/`FakeChatModel`.**

Reasons:
1. The thing Phases 3-4 actually build is *the translation between the wire and `TurnComplete`* — `AIMessageChunk` accumulation, `tool_call_chunks` → raw `arguments` strings, `usage_metadata` → `LLMUsage`, `response_metadata["finish_reason"]`. A fake chat model short-circuits every one of those. It would test nothing that changed.
2. The layer a fake chat model would occupy is **already occupied by `ScriptedLLM`**, which is cheaper (it stubs at `LLMClient`, above the model entirely) and is what the other ~70 tests use.
3. `MockTransport` keeps the SSE fixtures — real bytes from a real provider — as the regression corpus. Those fixtures are the only artefact in the repo that encodes what Groq actually sends.

Mechanically: `build_chat_model` calls `get_http_client()` from `app.agent.models` (re-exported from `llm_client` or moved there — either way, one accessor, patched by name, never the global, per `test_llm_client.py:32-34`'s stated rule).

### Sequencing so the suite is never red

| Step | Suite state |
|---|---|
| Phase 0 adds 3 test files against **current** code | green — new tests pass against the code they document |
| Phase 1 bumps deps, 2-line graph change | green |
| Phase 2 adds `models.py` + its tests, **zero importers** | green |
| Phase 3 rewrites `extract`; Seam A re-pointed in the *same commit* as the code it tests | green — Seams B, C, D untouched |
| Phase 4 rewrites `stream_chat`; `test_llm_client.py`'s `install()` re-pointed in the same commit | green — Seams B, C, D untouched |
| Phase 5 deletes dead code | green — nothing references it |
| Phase 6 tracing | green |

There is no window where the suite is red, because **no seam and no code move in different commits.**

### The gap Phase 0 exists to close, stated plainly

Today: `to_strict_json_schema` has no direct test; Seam C stubs `extract`, so the strict-schema construction at `llm_client.py:209-216` is **never exercised in CI**; `_backoff` and `_post`'s retry loop have no tests at all. A migration undertaken in that state is unfalsifiable — the suite would go green on a client that 400s on every real extraction. Phase 0 is therefore a **precondition**, not a nicety, and it is the one phase that must not be reordered.

---

## 5. Verification plan for the seven unknowns

| # | Unknown | Probe | Result that changes the plan |
|---|---|---|---|
| **a** | Do Groq / Gemini-compat / aicredits honour `"strict": true` on the wire? | Ad-hoc `scripts/probe_extract.py` (not committed): run `extract(schema=ExtractedJob, …)` on one real posting per provider, ×3 for stability. Assert a valid `ExtractedJob`, and that a deliberately over-constrained variant (add a `Literal` the posting cannot satisfy) is *refused* rather than hallucinated. **Run in Phase 3, before Phase 5.** | If a provider ignores `strict` → nothing changes structurally (we already parse and raise on `ValidationError`), but note it in `config.py`'s provider table so the graph's retry cap of 2 (`ingestion.py:36`) is understood as load-bearing for that provider. If a provider *rejects* the strict schema → keep that provider on the pre-migration `_post` path and do not delete `_post` in Phase 5. |
| **b** | Which `openai` major does the lockfile resolve? | **Resolved by decision, not discovery**: pin `openai>=1.99,<3` in Phase 1. Verify with `pip index versions openai`, `pip show openai`, and the committed tripwire test `assert langchain_openai.chat_models.base.httpx is httpx`. | If `langchain-openai>=1.6.0` hard-requires `openai>=3` (i.e. the `<3` pin is unsatisfiable), the plan changes materially: `http_async_client` can no longer take the existing pooled `httpx.AsyncClient`. Fallback — build the pool from `langchain_openai._compat.httpx` inside `models.py`, keep `close_http_client` pointed at it, and re-point Seam A's `MockTransport` to the httpx2 class. Adds ~20 LOC to Phase 2 and touches `test_llm_client.py`'s import line. **Check this first, in Phase 1, before writing Phase 2.** |
| **c** | Does an injected `http_async_client` disable or compose with the SDK's retry/timeout? | `models.py` sets `max_retries=0` and both timeouts consistently, so composition is moot. Confirm empirically: `MockTransport` counting requests, returning 429 twice then 200 — assert the transport saw **exactly** the number of attempts `settings.llm_max_retries` implies, no more. | If the SDK retries anyway despite `max_retries=0`, the effective retry count becomes N×M and the 120 s assistant deadline (`config.py:206`) is at risk. Mitigation: drop our loop to a single attempt and set `max_retries=settings.llm_max_retries`, accepting the loss of 413 retry — **and then explicitly add 413 back** via the transport-level `x-should-retry` behaviour or an outer `with_retry(retry_if_exception_type=(LLMError,))`. Document the trade. |
| **d** | Is `.parse()` (used whenever `response_format` is present, stricter than `.create()`) safe against non-OpenAI endpoints? | Same probe as (a), but assert on the raw path: log `type(exc)` on failure. Additionally, a `MockTransport` test returning a Groq-shaped body with extra top-level keys (`x_groq`, `queue_time`) — Groq does send these — and assert `extract` still succeeds. | If `.parse()` rejects a Groq/aicredits body → **do not migrate extraction at all**. Revert Phase 3, keep `_post` for `extract` only, and let Phase 5 delete only the streaming half. Phase 3 is deliberately separate from Phase 4 for exactly this outcome. |
| **e** | Streaming + `response_format` — does base.py:2137's `payload.pop("stream")` make them mutually exclusive? | Not exercised: extraction never streams (`stream_chat` never sets `response_format`). Confirm negatively with a `MockTransport` assertion that the extraction payload contains **no** `"stream"` key and the chat payload contains **no** `"response_format"` key. Committed as part of `test_llm_payload.py`. | Only matters if someone later wants streamed structured output. If they do, the answer is: they cannot on chat-completions, and the plan should say so in the docstring rather than have it discovered. |
| **f** | The exact langgraph 0.2.60 → 1.2.11 delta for **this** DAG | `python -m pytest tests/test_ingest.py -v` immediately after the Phase 1 install, before any code edit — that isolates the delta. Then `get_ingestion_graph().get_graph().draw_ascii()` diffed against the pre-bump output. Specifically check: the inline-lambda reducer at `:58` still applies; `session`/`client` as `Any` state fields (`:47-48`) still survive serialisation (v1 may be stricter about non-serialisable state); the inline-lambda conditional at `:213` still resolves; `config={"run_name","metadata"}` at `:260-267` still reaches LangSmith. | If v1 rejects non-serialisable state values, `session`/`client` move out of `IngestionState` into `context_schema` (`ainvoke(..., context=IngestionContext(session=…, client=…))`), and every node signature gains `runtime: Runtime[IngestionContext]`. That is a ~40 LOC change confined to `ingestion.py` — still Phase 1, but it doubles that phase's size. **This is the most likely of the seven to force a change.** |
| **g** | Does LangSmith's pricing table cover the Groq/Gemini model IDs under `ls_provider`? | After Phase 6, enable tracing against a scratch LangSmith project, run one extraction per provider, and read the cost column in the UI. Compare against the provider's own billing page for the same run. | If cost shows zero or wrong for a provider, the fix is a LangSmith custom model-pricing entry keyed on `ls_provider`+`ls_model_name` — configuration, not code. **But if it turns out only `ls_provider="openai"` prices at all**, the plan changes: keep `ls_provider` as the real provider (for grouping) and add a separate `ls_model_name` that LangSmith recognises, or accept unpriced runs and rely on `LLMUsage.total_tokens`. Do **not** flip `ls_provider` to `"openai"` — that silently reprices every historical run. |

---

## 6. Risk register

Ranked by (probability × silence). The top five all pass CI and break production.

| # | Risk | Trigger | Blast radius | Detection | Mitigation |
|---|---|---|---|---|---|
| **1** | 🔇 **`$ref` inlining lost** — someone "simplifies" the pinned dict into `with_structured_output(ExtractedJob)` | any future edit to `extract`; or a reviewer who reads only the LangChain docs | **Every** real extraction 400s: `/ingest` dead, `propose_tracked_posting` dead. CI stays green because Seam C stubs `extract` | none today | Phase 0's `test_extraction_schema.py` + `test_llm_payload.py` assert the exact outgoing dict and the absence of `$ref`. Phase 5's docstring states the reason. **This is the single highest-value test in the plan.** |
| **2** | 🔇 **`stream_usage` silently off** — `openai_api_base` is set *and* a custom http client is injected, and base.py:1332 auto-disables it on either | forgetting one kwarg in `build_chat_model` | Every streamed turn reports 0 tokens; `cache_hit_rate` → 0; LangSmith prices at zero; `AssistantResult.total_tokens` → 0. All three silent, exactly as `:270-273` warns | `test_chat_model.py::stream_usage is True` + `test_llm_payload.py` asserting `stream_options` on the wire | Explicit `stream_usage=True`, tested twice at two levels |
| **3** | 🔇 **`cache_read` lost** — langchain-openai's usage adapter may not map `prompt_tokens_details.cached_tokens` for a non-OpenAI base URL | provider response shape | Reported cost ≈ **double** what is billed (`tracing.py:148-150`); the per-round cache log (`assistant.py:239-244`) goes quiet | `test_tracing.py::test_usage_metadata_carries_cache_read` proves the *mapping*; only a live run proves the *provider* | Live probe in Phase 3 asserting non-zero `cached_tokens` on a second identical request. If absent, read `raw` response body via `include_raw`-equivalent or `response_metadata` and populate manually |
| **4** | 🔇 **`finish_reason=="length"` stops raising** | the check is 3 lines and easy to lose in a rewrite | The exact bug commit `9a7515f` fixed returns: truncated turn → empty content, no tool_calls → `assistant.py:247` reads "the model answered" → `:299-300` says *"I did not find anything to say about that"* — **while the user's turn is already committed** (`:302`) | `test_llm_client.py:65-96` (streamed) + **new** `test_extract_raises_on_truncation` (non-streamed, currently untested) | Both guards reimplemented explicitly on `response_metadata`; both under test before Phase 3 |
| **5** | 🔇 **`ls_provider` renamed to `"openai"`** by the nested `ChatOpenAI` run | doing nothing — this is the *default* | Historical cost attribution splits; every saved LangSmith filter on `ls_provider` stops matching; no error anywhere | `test_tracing.py` asserts the config value; only the LangSmith UI proves the outcome | Per-invocation `config={"metadata": {"ls_provider": settings.llm_provider}}` on every call (Phase 6) |
| **6** | **`ToolNode` concurrency against one RLS `AsyncSession`** | any future move to `create_agent`/`ToolNode` | `InterfaceError`/`IllegalStateChangeError` under multi-tool turns; or worse, pool exhaustion at `db_pool_size=10` | `test_agent_actions.py:492-511` would flake, not fail cleanly | **Not migrating the loop.** Documented as a non-goal in `assistant.py`'s docstring (Phase 7) so the reason survives the author |
| **7** | **`max_completion_tokens` → `max_tokens`** — `ChatOpenAI.max_tokens` maps to the deprecated wire field | using `max_tokens=` in `build_chat_model` | Groq's TPM pre-debit may not apply, or the ceiling is ignored → 8000 TPM budget blown → 413 storms | `test_llm_payload.py` asserts the field name | `max_tokens=None` + `.bind(max_completion_tokens=N)` |
| **8** | **`arguments` re-serialised from parsed `args`** instead of taken raw from `tool_call_chunks` | the obvious-looking implementation | `assistant.py:265-266`'s `JSONDecodeError → {}` branch becomes unreachable; a mid-JSON truncation now produces *plausible wrong arguments* instead of an empty call | `test_llm_client.py:120-149` asserts on the raw concatenated string | Take `full.tool_call_chunks[i]["args"]`; map `invalid_tool_calls` too |
| **9** | **`content` becomes `""` instead of `None`** on a tool-calls-only turn | `AIMessageChunk.content` defaults to `""` | `conftest.py`'s `calls()` shape and the real client diverge; the stub stops representing reality — the exact class of bug `conftest.py:313-314` was written to prevent | `test_llm_client.py:120-149` | explicit `or None` |
| **10** | **openai 3.x / httpx2** forces the pool out of `ChatOpenAI` | `<3` pin unsatisfiable, or lifted later | Seam A breaks; pooling, split timeout and loop-keying all need re-homing | `test_dependencies.py::test_langchain_and_openai_share_one_httpx` tripwire | Pin now; rebuild the pool from `langchain_openai._compat.httpx` if forced (unknown b) |
| **11** | **413 stops being retried** | trusting `max_retries` instead of `_backoff` | Groq TPM exhaustion surfaces as a hard `LLMError` on the first hit; `/ingest` failure rate rises under load | `test_llm_retry.py` | `max_retries=0` + our loop; asserted in Phase 0 and Phase 3 |
| **12** | **`openai.*` exception escapes `llm_client.py`** | a missed `except` branch | `LLMError` is not in `DOMAIN_ERROR_STATUS` (`main.py:62-66`), so anything else is a **500**: `/chat` loses its 422 (`agent.py:100`), `/chat/stream` loses its in-stream error frame (`:202`), and `matching.py:130-132`'s graceful 85% degrade becomes a hard failure | **new** `test_llm_retry.py::test_no_openai_exception_escapes` — parametrised over `APIStatusError`, `APIConnectionError`, `APITimeoutError`, `RateLimitError` | Catch-all mapping in `_invoke_with_retry` and `_stream` |
| **13** | **LangGraph v1 rejects non-serialisable state** (`session`/`client` as `Any`, `ingestion.py:47-48`) | Phase 1 install | Ingestion dead until moved to `context_schema` | `test_ingest.py` fails loudly at Phase 1 — **not silent** | Budgeted in Phase 1 (unknown f); ~40 LOC confined to `ingestion.py` |
| **14** | **Nested tracing doubles token counts in the tree** | Phase 6, by default | LangSmith aggregate cost for a project reads ~2× | visible in the UI | Outer `@traced` spans are `run_type="llm"` with `usage_metadata`; if double-counting appears, demote them to `run_type="chain"` (they keep their names and reshapers, lose the token column to the nested run) |
| **15** | **Tools payload uploaded per round** to LangSmith | Phase 6, by default | ~13 KB × 6 rounds × every turn of trace volume | LangSmith usage page | `Client(hide_inputs=…)` in `configure_tracing` — the only masking tier that covers auto-generated runs |
| **16** | **Deadlines untested** — `assistant_deadline_seconds=120`, `ingest_deadline_seconds=100` have no coverage (§7) | any retry-count change | An RLS transaction holds one of 10 pool connections for minutes (`config.py:201-205`) | none today | Out of scope, but add `test_llm_retry.py::test_total_retry_budget_fits_the_deadline` asserting `llm_max_retries × (timeout + max backoff) < ingest_deadline_seconds` — a cheap arithmetic guard |

---

## 7. Estimated size and commit sequence

| Phase | Files | Prod LOC Δ | Test LOC Δ | Effort |
|---|---|---|---|---|
| 0 — golden tests | 3 new (tests) | 0 | +260 | half day |
| 1 — deps + LangGraph 1.x | 3 | +25 / −8 | +30 | half day (2 days if unknown *f* bites) |
| 2 — chat-model factory | 2 new | +90 | +110 | half day |
| 3 — extraction path | 1 | +60 / −45 | +50 | 1 day + live probes |
| 4 — chat/stream path | 1 | +85 / −110 | ~15 changed | 1–1.5 days |
| 5 — delete wire layer | 1 | −180 | 0 | 1 hour |
| 6 — tracing | 2 | +30 / −5 | +45 | half day |
| 7 — docs | 4 | ~+40 (comments) | 0 | 1 hour |
| **Total** | **~14** | **≈ −8 net** (~−180 deleted, ~+175 added) | **+495** | **≈ 5–6 days** |

Net production LOC is roughly flat, which is the honest picture: this migration buys **correctness of the wire layer** (SSE framing, chunk accumulation, provider-maintained parsing) and **future optionality**, not brevity. The retry policy, the backoff, the error taxonomy, the schema transform, the tool schemas and the loop all stay, because each encodes something the framework does not know about this deployment.

### Commit sequence

Direct to `main`, repo convention `type(scope): imperative clause, and second clause`, no trailer:

```
1  test(llm): pin the strict schema and the wire payload before they move
2  test(llm): cover the retry loop, the 413 reading, and the backoff jitter
3  chore(deps): move to langgraph 1.x, and add langchain-core and langchain-openai
4  feat(llm): build the chat model from the provider table, and keep the pooled client
5  refactor(llm): extract through ChatOpenAI, and keep the pinned response_format
6  refactor(llm): stream through ChatOpenAI, and keep the TurnComplete contract
7  refactor(llm): delete the hand-rolled SSE reader, and keep the retry policy
8  fix(tracing): name the real provider on the nested run, and keep the cache read
9  docs(agent): say why the loop is not an agent, and why the tools are not @tool
```

Commits 1–2 are shippable and useful on their own even if the migration is abandoned. Commit 3 is the only irreversible-ish one (dependency floor); commits 5 and 6 are independently revertible, which is deliberate — unknown (d) could kill 5 without touching 6, and unknown (e)/(a) could kill 6 without touching 5. Commit 7 must not land until both 5 and 6 have been exercised live against all three providers.

---

### Critical files for implementation
- `D:\projects\AI_Job_Tracker\backend\app\agent\llm_client.py`
- `D:\projects\AI_Job_Tracker\backend\app\schemas\extraction.py`
- `D:\projects\AI_Job_Tracker\backend\tests\conftest.py`
- `D:\projects\AI_Job_Tracker\backend\app\agent\graphs\ingestion.py`
- `D:\projects\AI_Job_Tracker\backend\pyproject.toml`
---

## 8. Outcome — what the unknowns turned out to be

Recorded after the fact, because five of the seven were resolved by measurement
rather than by reading, and the answers are what the code now depends on.

| # | Unknown | Resolved as |
|---|---|---|
| **a** | Do the providers honour `"strict": true`? | **Yes, on aicredits.** A live extraction of a real posting returned a valid `ExtractedJob` twice, with the schema pinned. Groq and Gemini remain unprobed. |
| **b** | Which `openai` major resolves? | **2.54.0**, so `http_async_client` still takes a plain `httpx.AsyncClient` and the pool, the split timeout and the loop keying all carried across unchanged. Pinned `<3` and guarded by `tests/test_dependencies.py`. Note `httpx2` is now installed anyway as a langsmith dependency — the pin is about which one the *SDK* uses, and the test asserts exactly that. |
| **c** | Does an injected client disable the SDK's retries? | **Moot.** `max_retries=0` and one timeout value make the composition question unnecessary; `tests/test_llm_retry.py` counts the attempts that reach the transport. |
| **d** | Is `.parse()` safe against non-OpenAI bodies? | **Yes**, on Groq-shaped and aicredits bodies, both in tests and live. It also surfaced something the plan did not predict — see below. |
| **e** | Streaming plus `response_format`? | **Mutually exclusive, and neither path wants both.** Extraction now sends `"stream": false` explicitly where it used to omit the field; asserted in `test_llm_payload.py`. |
| **f** | The langgraph 0.2 → 1.2 delta for this DAG? | **Already paid.** The container had been resolving 1.2.11 under a `>=0.2.60` floor for some time and the suite was green. Only `set_entry_point` → `add_edge(START, …)` changed. The topology is now asserted so the next major is a decision. |
| **g** | Does LangSmith price these model ids? | **Still open.** Needs a live traced run; `ls_provider` is set to the real provider so the question is about the catalogue, not the metadata. |

### The thing the plan missed

`openai.LengthFinishReasonError`. When a `response_format` is set, the SDK
notices truncation before any of our code does — and raises something that is
deliberately *not* an `APIError`, so it passed every handler and left the module
uncaught. `LLMError` is absent from the domain-error status table, so that would
have been a 500 in place of `/chat`'s 422, the in-stream error frame on
`/chat/stream`, and the graceful score-from-85% degrade in matching. It is
mapped now, and the explicit `finish_reason` check stays for the paths that set
no response format.

### Risk #3 closed by measurement

`cache_read` was the risk rated most likely to silently double the reported
cost. A live second extraction of the same posting reported `cached=1664` of
1699 prompt tokens, so Groq's `prompt_tokens_details.cached_tokens` does survive
LangChain's normalisation on a non-OpenAI host. Covered by `test_tracing.py` at
three levels, including one streamed turn end to end.
