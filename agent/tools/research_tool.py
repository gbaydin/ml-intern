"""
Research subagent tool — spawns a cheap LLM call with a focused
research task and returns a summary. The subagent gets its own
independent context (not the main conversation), so research
work doesn't pollute the main agent's context window.

Inspired by claude-code's code-explorer agent pattern.
"""

import json
import logging
from typing import Any

from litellm import Message, acompletion

from agent.core.doom_loop import check_for_doom_loop
from agent.core.llm_params import _resolve_llm_params
from agent.core.session import Event

logger = logging.getLogger(__name__)

# Context budget for the research subagent (tokens).
# When usage exceeds WARN threshold, the subagent is told to wrap up.
# At MAX, the loop is force-stopped and whatever content exists is returned.
_RESEARCH_CONTEXT_WARN = 170_000  # 85% of 200k
_RESEARCH_CONTEXT_MAX = 190_000

# Tools the research agent can use (read-only subset)
RESEARCH_TOOL_NAMES = {
    "read",
    "bash",
    "web_search",
    "explore_hf_docs",
    "fetch_hf_docs",
    "find_hf_api",
    "hf_papers",
    "github_find_examples",
    "github_list_repos",
    "github_read_file",
    "hf_inspect_dataset",
    "hf_repo_files",
}

RESEARCH_SYSTEM_PROMPT = """\
You are a research sub-agent for an ML engineering assistant.
You handle TWO types of research tasks:

1. **Literature crawls**: Find training recipes, evaluation protocols, SOTA methods
2. **Implementation research**: Solve specific technical questions — how to use a library API, compute a metric, install a package, work around a missing dependency

Read the task description carefully and pick the right approach.

# For literature crawls (training recipes, SOTA comparison)

Start from papers. Papers contain the results, and results tell you what works.

## The crawl

1. **Find anchor papers**: Search for the task/domain. Identify the landmark paper(s).
2. **Crawl the citation graph**: Look DOWNSTREAM (papers that cite it) — these improved on it.
3. **Read methodology sections**: Use `read_paper` sections 3, 4, 5 (Methodology, Experiments, Results). Extract exact datasets, training configs, and results.
4. **Attribute results to recipes**: Every finding must link a RESULT to the RECIPE that produced it.
5. **Validate datasets**: Check HF Hub with `hf_inspect_dataset`.
6. **Find code**: Use `github_find_examples` and `github_read_file`.

## When to go deeper

- If the anchor paper is old (>1 year), its citation graph is your main source.
- If a downstream paper reports significantly better results, crawl ITS citation graph too.
- Use `snippet_search` to find specific claims across papers.

# For implementation research (technical questions)

When the task is about HOW to do something (compute a metric, use a library, fix a dependency), use a different strategy:

1. **Web search first**: Use `web_search` to find documentation, StackOverflow answers, tutorials, and code examples for the specific question.
2. **Fetch documentation pages**: Use `web_search` with `fetch_content=true` to read the actual documentation.
3. **Check papers for methodology**: If the question involves a scientific metric or evaluation protocol, search papers to find how others computed it. Read their methodology sections for implementation details.
4. **Find code on GitHub**: Use `github_find_examples` and `github_read_file` to find working implementations.
5. **Check package availability**: Use `bash` to run `pip index versions <package>` or `pip show <package>` to check if packages are installable.

CRITICAL: Do not give up after one failed attempt. If a package isn't installed, check if it can be pip-installed. If an API isn't available, search for alternatives. If you can't find a direct solution, look for how others solved the same problem in their code.

STAY FOCUSED: Limit yourself to 2-3 web searches per task. Extract the specific answer you need (code snippet, API call, package name) and move on. Do not follow tangential links or explore interesting-but-irrelevant results. Your job is to return a concise, actionable answer, not an exhaustive survey.

## Example: "How to compute E_hull without Materials Project API"

```
# 1. Web search for the specific question
web_search({"query": "pymatgen compute energy above hull from local data PhaseDiagram ComputedEntry"})

# 2. Search papers for methodology
hf_papers({"operation": "snippet_search", "query": "self-consistent convex hull MLIP energy above hull crystal generation"})

# 3. Read a relevant paper's methodology
hf_papers({"operation": "read_paper", "arxiv_id": "...", "section": "3"})

# 4. Find code examples on GitHub
web_search({"query": "pymatgen PhaseDiagram ComputedEntry example site:github.com"})

# 5. Check if the needed package is installable
bash({"command": "pip index versions mp-api 2>&1 | head -5"})
```

# How to use your tools

## Web search (USE FOR IMPLEMENTATION QUESTIONS)
- `web_search(query=...)`: General web search — documentation, StackOverflow, tutorials, library APIs
- `web_search(query=..., fetch_content=true)`: Search AND fetch full page content from top results

## Papers & citations (USE FOR LITERATURE CRAWLS)
- `hf_papers(operation="search", query=...)`: Search papers
- `hf_papers(operation="search", query=..., min_citations=50, sort_by="citationCount")`: Highly-cited papers
- `hf_papers(operation="search", query=..., date_from="2024-01-01")`: Recent papers
- `hf_papers(operation="paper_details", arxiv_id=...)`: Metadata, citations, TL;DR
- `hf_papers(operation="citation_graph", arxiv_id=...)`: References + citing papers
- `hf_papers(operation="read_paper", arxiv_id=..., section="3")`: Read a section's full text
- `hf_papers(operation="read_paper", arxiv_id=...)`: Get TOC (section list)
- `hf_papers(operation="snippet_search", query=...)`: Semantic search across 12M+ paper passages
- `hf_papers(operation="recommend", arxiv_id=...)`: Find related papers
- `hf_papers(operation="find_datasets", arxiv_id=...)`: HF datasets linked to a paper
- `hf_papers(operation="find_all_resources", arxiv_id=...)`: Datasets + models + collections

## Dataset inspection
- `hf_inspect_dataset`: Check dataset schema, splits, sample rows

## GitHub code research
- `github_find_examples`: Find example scripts in HF repos
- `github_read_file`: Read implementation code

## Documentation
- `explore_hf_docs(endpoint)`: Search HF library docs
- `fetch_hf_docs(url)`: Fetch full page content from explore results
- `find_hf_api(query=..., tag=...)`: Find REST API endpoints

## Hub repo inspection
- `hf_repo_files`: List/read files in any HF repo

# Output format

Adapt your output to the task type:

## For literature crawls: Recipe table

For each promising approach, report:
- **Paper**: title, arxiv_id, date, venue
- **Result**: exact benchmark scores
- **Dataset(s)**: name, size, HF Hub availability
- **Method**: training approach, key hyperparameters
- **What made it work**: the specific insight

Also include SOTA landscape, essential references, and code patterns.

## For implementation research: Direct answers

- **Solution**: The specific approach that works, with code
- **Alternatives tried**: What doesn't work and why
- **Code example**: Working code snippet the main agent can use directly
- **Dependencies**: What needs to be installed (`pip install X`)
- **Key references**: Documentation URLs, relevant paper sections

Be concise. Your output goes into another agent's context — every token counts.
Aim for 500-1500 words max. Include actual code snippets, not paraphrased descriptions.
"""

RESEARCH_TOOL_SPEC = {
    "name": "research",
    "description": (
        "Spawn a research sub-agent to investigate questions WITHOUT polluting "
        "the main conversation context. The sub-agent gets its own independent "
        "context window with research tools and returns a concise summary.\n\n"
        "Use this for TWO types of tasks:\n\n"
        "1. LITERATURE CRAWLS: Finding training recipes, SOTA methods, "
        "evaluation protocols, benchmark comparisons — anything where you "
        "need to read papers and trace citation graphs.\n\n"
        "2. IMPLEMENTATION RESEARCH: Solving technical questions — how to "
        "use a library API, compute a specific metric, work around a missing "
        "dependency, find working code examples. The sub-agent has web_search "
        "for general web queries (documentation, StackOverflow, tutorials, "
        "library APIs like pymatgen, ASE, scikit-learn, etc.).\n\n"
        "IMPORTANT: When you hit a technical blocker (missing package, "
        "unfamiliar API, don't know how to compute something), use this tool "
        "to research the solution BEFORE falling back to a proxy or giving up.\n\n"
        "Available sub-agent tools: web_search, hf_papers, github_find_examples, "
        "github_read_file, explore_hf_docs, fetch_hf_docs, hf_inspect_dataset, "
        "hf_repo_files, bash, read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Detailed description of what to research. Be specific: "
                    "include library names, function names, error messages, "
                    "metric names, or arxiv IDs when you have them.\n"
                    "Examples:\n"
                    "- 'Research how to compute energy above hull (E_hull) "
                    "using pymatgen PhaseDiagram from local structure data "
                    "without Materials Project API access'\n"
                    "- 'Find current TRL SFTTrainer usage: working example "
                    "scripts, documentation, SFTConfig parameters'\n"
                    "- 'Literature crawl for crystal structure generation. "
                    "Start from CDVAE. Find SOTA evaluation protocols for "
                    "SUN metrics.'"
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional context from the current conversation that the "
                    "research agent needs (e.g., what the user wants to build, "
                    "constraints, what's been tried, what failed)."
                ),
            },
        },
        "required": ["task"],
    },
}


def _get_research_model(main_model: str) -> str:
    """Pick a cheaper model for research based on the main model."""
    if "anthropic/" in main_model:
        return "anthropic/claude-sonnet-4-6"
    # For non-Anthropic models (HF router etc.), use the same model
    return main_model


async def research_handler(
    arguments: dict[str, Any], session=None, tool_call_id: str | None = None, **_kw
) -> tuple[str, bool]:
    """Execute a research sub-agent with its own context."""
    task = arguments.get("task", "")
    context = arguments.get("context", "")
    if not task:
        return "No research task provided.", False

    if not session:
        return "No session available for research agent.", False

    # Build the sub-agent's messages (independent context)
    messages: list[Message] = [
        Message(role="system", content=RESEARCH_SYSTEM_PROMPT),
    ]

    user_content = f"Research task: {task}"
    if context:
        user_content = f"Context: {context}\n\n{user_content}"
    messages.append(Message(role="user", content=user_content))

    # Use a cheaper/faster model for research
    main_model = session.config.model_name
    research_model = _get_research_model(main_model)
    # Research is a cheap sub-call — cap the main session's effort at "high"
    # so a user preference of ``max``/``xhigh`` (valid for Opus 4.6/4.7) doesn't
    # propagate to a Sonnet research model that may not accept those levels.
    # We also haven't probed this sub-model so we don't know its ceiling.
    _pref = getattr(session.config, "reasoning_effort", None)
    _capped = "high" if _pref in ("max", "xhigh") else _pref
    llm_params = _resolve_llm_params(
        research_model,
        getattr(session, "hf_token", None),
        reasoning_effort=_capped,
    )

    # Get read-only tool specs from the session's tool router
    tool_specs = [
        spec
        for spec in session.tool_router.get_tool_specs_for_llm()
        if spec["function"]["name"] in RESEARCH_TOOL_NAMES
    ]

    # Unique ID + short label so parallel agents show separate status lines.
    # Use the tool_call_id when available — it's unique per invocation and lets
    # the frontend match a research tool card to its agent state. Fall back to
    # uuid for offline/test paths. Previously used md5(task), which collided
    # when the same task string was researched in parallel.
    if tool_call_id:
        _agent_id = tool_call_id
    else:
        import uuid
        _agent_id = uuid.uuid4().hex[:8]
    _agent_label = "research: " + (task[:50] + "…" if len(task) > 50 else task)

    async def _log(text: str) -> None:
        """Send a progress event to the UI so it doesn't look frozen."""
        try:
            await session.send_event(
                Event(event_type="tool_log", data={
                    "tool": "research",
                    "log": text,
                    "agent_id": _agent_id,
                    "label": _agent_label,
                })
            )
        except Exception:
            pass

    _tool_uses = 0
    _total_tokens = 0
    _warned_context = False

    await _log("Starting research sub-agent...")

    # Run the research loop — context budget is the real limiter
    max_iterations = 60
    for _iteration in range(max_iterations):
        # ── Doom-loop detection ──
        doom_prompt = check_for_doom_loop(messages)
        if doom_prompt:
            logger.warning("Research sub-agent doom loop detected at iteration %d", _iteration)
            await _log("Doom loop detected — injecting corrective prompt")
            messages.append(Message(role="user", content=doom_prompt))

        # ── Context budget: warn at 75%, hard-stop at 95% ──
        if _total_tokens >= _RESEARCH_CONTEXT_MAX:
            logger.warning(
                "Research sub-agent hit context max (%d tokens) — forcing summary",
                _total_tokens,
            )
            await _log(f"Context limit reached ({_total_tokens} tokens) — forcing wrap-up")
            # Ask for a final summary with no tools
            messages.append(Message(
                role="user",
                content=(
                    "[SYSTEM: CONTEXT LIMIT REACHED] You have used all available context. "
                    "Summarize your findings NOW. Do NOT call any more tools."
                ),
            ))
            try:
                response = await acompletion(
                    messages=messages,
                    tools=None,  # no tools — force text response
                    stream=False,
                    timeout=120,
                    **llm_params,
                )
                content = response.choices[0].message.content or ""
                return content or "Research context exhausted — no summary produced.", bool(content)
            except Exception:
                return "Research context exhausted and summary call failed.", False

        if not _warned_context and _total_tokens >= _RESEARCH_CONTEXT_WARN:
            _warned_context = True
            await _log(f"Context at {_total_tokens} tokens — nudging to wrap up")
            messages.append(Message(
                role="user",
                content=(
                    "[SYSTEM: You have used 75% of your context budget. "
                    "Start wrapping up: finish any critical lookups, then "
                    "produce your final summary within the next 1-2 iterations.]"
                ),
            ))

        try:
            response = await acompletion(
                messages=messages,
                tools=tool_specs if tool_specs else None,
                tool_choice="auto",
                stream=False,
                timeout=120,
                **llm_params,
            )
        except Exception as e:
            logger.error("Research sub-agent LLM error: %s", e)
            return f"Research agent LLM error: {e}", False

        # Track tokens
        if response.usage:
            _total_tokens = response.usage.total_tokens
            await _log(f"tokens:{_total_tokens}")

        choice = response.choices[0]
        msg = choice.message

        # If no tool calls, we have our final answer
        if not msg.tool_calls:
            await _log("Research complete.")
            content = msg.content or "Research completed but no summary generated."
            return content, True

        # Execute tool calls and add results.
        # Rebuild the assistant message with only the wire-safe fields —
        # LiteLLM's raw Message carries `provider_specific_fields` and
        # `reasoning_content`, which the HF router's OpenAI schema rejects
        # if we echo them back in the next request.
        messages.append(Message(
            role="assistant",
            content=msg.content,
            tool_calls=msg.tool_calls,
        ))
        for tc in msg.tool_calls:
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                messages.append(
                    Message(
                        role="tool",
                        content="Invalid tool arguments.",
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    )
                )
                continue

            tool_name = tc.function.name
            if tool_name not in RESEARCH_TOOL_NAMES:
                messages.append(
                    Message(
                        role="tool",
                        content=f"Tool '{tool_name}' not available for research.",
                        tool_call_id=tc.id,
                        name=tool_name,
                    )
                )
                continue

            try:
                import json as _json

                args_str = _json.dumps(tool_args)[:80]
                await _log(f"▸ {tool_name}  {args_str}")

                output, _success = await session.tool_router.call_tool(
                    tool_name, tool_args, session=session
                )
                _tool_uses += 1
                await _log(f"tools:{_tool_uses}")
                # Truncate tool output for the research context
                if len(output) > 8000:
                    output = output[:4800] + "\n...(truncated)...\n" + output[-3200:]
            except Exception as e:
                output = f"Tool error: {e}"

            messages.append(
                Message(
                    role="tool",
                    content=output,
                    tool_call_id=tc.id,
                    name=tool_name,
                )
            )

    # ── Iteration limit: try to salvage findings ──
    await _log("Iteration limit reached — extracting summary")
    messages.append(Message(
        role="user",
        content=(
            "[SYSTEM: ITERATION LIMIT] You have reached the maximum number of research "
            "iterations. Summarize ALL findings so far. Do NOT call any more tools."
        ),
    ))
    try:
        response = await acompletion(
            messages=messages,
            tools=None,
            stream=False,
            timeout=120,
            **llm_params,
        )
        content = response.choices[0].message.content or ""
        if content:
            return content, True
    except Exception as e:
        logger.error("Research summary call failed: %s", e)

    return (
        "Research agent hit iteration limit (60). "
        "Partial findings may be incomplete — try a more focused task.",
        False,
    )
