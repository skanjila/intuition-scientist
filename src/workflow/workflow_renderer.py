"""Workflow visibility renderer.

Produces the **Deep workflow-visibility output** section that helps users
understand what happened inside the agentic pipeline for each request.

Three helpers are exposed:

* :func:`build_workflow_trace` — build a :class:`~src.models.WorkflowTrace`
  from a completed :class:`~src.models.WeighingResult`.
* :func:`render_workflow` — render a :class:`~src.models.WorkflowTrace` to a
  plain-text string according to the selected
  :class:`~src.models.WorkflowMapMode`.

No raw chain-of-thought is disclosed; only an explainability summary is
produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models import WorkflowMapMode, WorkflowStep, WorkflowTrace

if TYPE_CHECKING:
    from src.models import WeighingResult

# Stable heading that tests and downstream tools can grep for
WORKFLOW_HEADING = "## Workflow"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_workflow_trace(result: "WeighingResult", *, use_mcp: bool = False) -> WorkflowTrace:
    """Build a :class:`WorkflowTrace` from a completed :class:`WeighingResult`.

    The trace is assembled *after* the run so that no hidden chain-of-thought
    is leaked — only observable facts (domains queried, tool usage, scores) are
    recorded.
    """
    domains_queried = [r.domain.value.replace("_", " ").title() for r in result.agent_responses]
    n_domains = len(domains_queried)
    mcp_used = use_mcp and any(r.mcp_context for r in result.agent_responses)

    # ------------------------------------------------------------------ steps
    steps: list[WorkflowStep] = [
        WorkflowStep(
            label="Receive question",
            description="User question ingested and forwarded to the orchestrator.",
        ),
    ]
    if mcp_used:
        steps.append(WorkflowStep(
            label="MCP web search",
            tool="mcp_search",
            description="DuckDuckGo web-search queries issued for additional evidence.",
            tool_result_summary=f"Retrieved web context for {n_domains} domain(s).",
        ))
    steps.append(WorkflowStep(
        label="Domain routing",
        description=(
            f"Question routed to {n_domains} domain agent(s): "
            + ", ".join(domains_queried[:5])
            + (" …" if n_domains > 5 else "")
        ),
    ))
    steps.append(WorkflowStep(
        label="Dual-pipeline inference",
        description=(
            "Each agent ran the intuition path (knowledge only) "
            "and the tool/MCP path, then blended them with learned weights."
        ),
    ))
    steps.append(WorkflowStep(
        label="Alignment scoring",
        description=(
            "Semantic similarity between human intuition and each agent "
            "answer computed (Jaccard + TF-IDF)."
        ),
    ))
    steps.append(WorkflowStep(
        label="Synthesis",
        description="Weighted consensus answer synthesised across all domain experts.",
    ))
    steps.append(WorkflowStep(
        label="Output",
        description="Final result returned to the user.",
    ))

    # ---------------------------------------------------------- inputs_context
    inputs_context: list[str] = [
        f'Question: "{result.question}"',
        f"Domains queried: {', '.join(domains_queried) or 'auto-detected'}",
        f"MCP/web search: {'enabled' if mcp_used else 'disabled'}",
        f"Human intuition confidence: {result.human_intuition.confidence:.0%}",
    ]

    # ------------------------------------------------------------ assumptions
    assumptions: list[str] = [
        "Human intuition was provided before agent inference (no leakage).",
        "All agent responses use the same LLM backend for consistency.",
        "Semantic similarity is an approximation; it may miss nuanced agreement.",
    ]
    if not mcp_used:
        assumptions.append("MCP was disabled; results rely solely on model knowledge.")

    # ------------------------------------------------------------------- plan
    plan: list[str] = [
        "Capture human intuitive answer and confidence.",
        f"Route question to {n_domains} relevant domain agent(s).",
        "Run dual-pipeline (intuition + tool evidence) for each agent.",
        "Score semantic alignment between human intuition and agent answers.",
        "Synthesise a weighted consensus answer.",
        "Compute overall intuition accuracy score.",
        "Return structured WeighingResult with recommendations.",
    ]

    # -------------------------------------------------------------- tool calls
    tool_calls: list[tuple[str, str, str]] = []
    if mcp_used:
        tool_calls.append((
            "mcp_search",
            "Retrieve up-to-date web evidence to ground agent answers.",
            f"Web context retrieved for {n_domains} domain(s); used in tool-path inference.",
        ))

    # --------------------------------------------------- intermediate_artifacts
    # Build a simple alignment table
    table_lines = ["| Domain | Similarity | Agent Confidence |", "|---|---|---|"]
    for align, resp in zip(result.alignment_scores, result.agent_responses):
        domain_label = align.domain.value.replace("_", " ").title()
        table_lines.append(
            f"| {domain_label} | {align.semantic_similarity:.2f} | {resp.confidence:.0%} |"
        )
    alignment_table = "\n".join(table_lines)

    accuracy_pct = result.intuition_accuracy_pct
    acceptance = (
        f"- [{'x' if accuracy_pct >= 70 else ' '}] Intuition accuracy ≥ 70 %  "
        f"(actual: {accuracy_pct:.1f} %)\n"
        f"- [{'x' if accuracy_pct >= 40 else ' '}] Intuition accuracy ≥ 40 %\n"
        f"- [x] Synthesised answer produced\n"
        f"- [x] Recommendations generated ({len(result.recommendations)} item(s))"
    )

    intermediate_artifacts: list[str] = [
        f"**Domain alignment table:**\n\n{alignment_table}",
        f"**Acceptance criteria:**\n\n{acceptance}",
    ]

    # -------------------------------------------------------------- next actions
    next_actions: list[str] = [
        "Try `--domains <domain>` to drill into a specific area.",
        "Run with `--use-mcp` to add live web evidence." if not mcp_used else "MCP already enabled.",
        "Use `--workflow-map deep` to see this full breakdown every time.",
        "Run `--fast` for a lower-latency preset (fewer domains, no MCP).",
    ]

    return WorkflowTrace(
        question=result.question,
        steps=steps,
        inputs_context=inputs_context,
        assumptions=assumptions,
        plan=plan,
        tool_calls=tool_calls,
        intermediate_artifacts=intermediate_artifacts,
        next_actions=next_actions,
    )


def render_workflow(trace: WorkflowTrace, mode: WorkflowMapMode) -> str:
    """Render *trace* to a plain-text string for the selected *mode*.

    Returns an empty string when *mode* is :attr:`~WorkflowMapMode.OFF`.
    """
    if mode is WorkflowMapMode.OFF:
        return ""

    lines: list[str] = []

    if mode is WorkflowMapMode.COMPACT:
        lines.append(f"{WORKFLOW_HEADING} (Compact)")
        lines.append("")
        lines.append(_mermaid_diagram(trace))
        return "\n".join(lines)

    if mode is WorkflowMapMode.STANDARD:
        lines.append(f"{WORKFLOW_HEADING} (Standard)")
        lines.append("")
        lines.append(_mermaid_diagram(trace))
        lines.append("")
        lines.append("### Inputs & context")
        for item in trace.inputs_context:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### Plan")
        for i, step in enumerate(trace.plan, 1):
            lines.append(f"{i}. {step}")
        return "\n".join(lines)

    # --- DEEP ---
    lines.append(f"{WORKFLOW_HEADING} (Deep)")
    lines.append("")

    # (A) Mermaid diagram
    lines.append("### (A) Mermaid workflow diagram")
    lines.append("")
    lines.append(_mermaid_diagram(trace))
    lines.append("")

    # (B) Inputs & context
    lines.append("### (B) Inputs & context")
    for item in trace.inputs_context:
        lines.append(f"- {item}")
    lines.append("")

    # (C) Assumptions
    lines.append("### (C) Assumptions")
    for assumption in trace.assumptions:
        lines.append(f"- {assumption}")
    lines.append("")

    # (D) Plan
    lines.append("### (D) Plan")
    for i, step in enumerate(trace.plan, 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    # (E) Tool-call plan & results
    lines.append("### (E) Tool-call plan & results")
    if trace.tool_calls:
        for tool_name, reason, result_summary in trace.tool_calls:
            lines.append(f"- **{tool_name}**: {reason}")
            lines.append(f"  - Result: {result_summary}")
    else:
        lines.append("- No external tool calls were made for this request.")
    lines.append("")

    # (F) Intermediate artifacts
    lines.append("### (F) Intermediate artifacts")
    for artifact in trace.intermediate_artifacts:
        lines.append(artifact)
        lines.append("")

    # (G) Next actions
    lines.append("### (G) Next actions / options")
    for action in trace.next_actions:
        lines.append(f"- {action}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_mermaid_label(label: str) -> str:
    """Escape characters that would break a Mermaid node label string."""
    # Replace backslashes first to avoid double-escaping
    label = label.replace("\\", "\\\\")
    # Replace double-quotes with single-quotes (Mermaid uses double-quoted strings)
    label = label.replace('"', "'")
    # Square brackets would open/close Mermaid node syntax; replace with parens
    label = label.replace("[", "(").replace("]", ")")
    # Replace literal newlines with Mermaid's line-break escape
    label = label.replace("\n", "\\n")
    return label


def _mermaid_diagram(trace: WorkflowTrace) -> str:
    """Return a fenced Mermaid ``flowchart TD`` block for *trace*."""
    lines: list[str] = ["```mermaid", "flowchart TD"]

    node_ids: list[str] = []
    for i, step in enumerate(trace.steps):
        node_id = f"S{i}"
        node_ids.append(node_id)
        label = _sanitize_mermaid_label(step.label)
        if step.tool:
            tool = _sanitize_mermaid_label(step.tool)
            lines.append(f'  {node_id}["{label}\\n🔧 {tool}"]')
        else:
            lines.append(f'  {node_id}["{label}"]')

    # Chain nodes sequentially
    for a, b in zip(node_ids, node_ids[1:]):
        lines.append(f"  {a} --> {b}")

    lines.append("```")
    return "\n".join(lines)
