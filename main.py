#!/usr/bin/env python3
"""Human Intuition Scientist — CLI entry point.

Usage
-----
    python main.py                               # fully interactive (mock backend)
    python main.py --question "..."              # supply question on the CLI
    python main.py --provider ollama:llama3.1:8b # use Ollama local model
    python main.py --provider groq:llama-3.1-8b-instant  # use Groq free-tier
    python main.py --no-mcp                      # disable internet search
    python main.py --domains physics cs          # restrict to specific domains
    python main.py --fast                        # lowest-latency preset (Ollama/Apple Silicon)
    python main.py --max-workers 2               # custom thread-pool size

Supported free/open backends:  mock, ollama, llamacpp, groq, together,
                                cloudflare, openrouter.
Anthropic and OpenAI are NOT supported (paid/proprietary providers).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Try to import Rich for pretty output; fall back to plain print if missing
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _console = Console()

    def _print(msg: str = "", **kw: object) -> None:
        _console.print(msg, **kw)  # type: ignore[arg-type]

    def _rule(title: str = "") -> None:
        _console.rule(title)

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    def _print(msg: str = "", **kw: object) -> None:  # type: ignore[misc]
        print(msg)

    def _rule(title: str = "") -> None:  # type: ignore[misc]
        print(f"\n{'='*70} {title}")


# ---------------------------------------------------------------------------
# Domain name mapping for CLI flags
# ---------------------------------------------------------------------------
from src.models import Domain, WeighingResult, WorkflowMapMode

_DOMAIN_MAP: dict[str, Domain] = {
    # Core science / engineering
    "ee": Domain.ELECTRICAL_ENGINEERING,
    "electrical_engineering": Domain.ELECTRICAL_ENGINEERING,
    "cs": Domain.COMPUTER_SCIENCE,
    "computer_science": Domain.COMPUTER_SCIENCE,
    "nn": Domain.NEURAL_NETWORKS,
    "neural_networks": Domain.NEURAL_NETWORKS,
    "social": Domain.SOCIAL_SCIENCE,
    "social_science": Domain.SOCIAL_SCIENCE,
    "space": Domain.SPACE_SCIENCE,
    "space_science": Domain.SPACE_SCIENCE,
    "physics": Domain.PHYSICS,
    "dl": Domain.DEEP_LEARNING,
    "deep_learning": Domain.DEEP_LEARNING,
    # High-economic-value industry domains
    "healthcare": Domain.HEALTHCARE,
    "climate": Domain.CLIMATE_ENERGY,
    "climate_energy": Domain.CLIMATE_ENERGY,
    "finance": Domain.FINANCE_ECONOMICS,
    "finance_economics": Domain.FINANCE_ECONOMICS,
    "economics": Domain.FINANCE_ECONOMICS,
    "cybersecurity": Domain.CYBERSECURITY,
    "cyber": Domain.CYBERSECURITY,
    "biotech": Domain.BIOTECH_GENOMICS,
    "genomics": Domain.BIOTECH_GENOMICS,
    "biotech_genomics": Domain.BIOTECH_GENOMICS,
    "supply_chain": Domain.SUPPLY_CHAIN,
    "logistics": Domain.SUPPLY_CHAIN,
    # Enterprise problem domains
    "legal": Domain.LEGAL_COMPLIANCE,
    "legal_compliance": Domain.LEGAL_COMPLIANCE,
    "compliance": Domain.LEGAL_COMPLIANCE,
    "architecture": Domain.ENTERPRISE_ARCHITECTURE,
    "enterprise_architecture": Domain.ENTERPRISE_ARCHITECTURE,
    "marketing": Domain.MARKETING_GROWTH,
    "marketing_growth": Domain.MARKETING_GROWTH,
    "growth": Domain.MARKETING_GROWTH,
    "org": Domain.ORGANIZATIONAL_BEHAVIOR,
    "organizational_behavior": Domain.ORGANIZATIONAL_BEHAVIOR,
    "hr": Domain.ORGANIZATIONAL_BEHAVIOR,
    "strategy": Domain.STRATEGY_INTELLIGENCE,
    "strategy_intelligence": Domain.STRATEGY_INTELLIGENCE,
    "intel": Domain.STRATEGY_INTELLIGENCE,
    # Mastery / interview / PhD research domains
    "algorithms": Domain.ALGORITHMS_PROGRAMMING,
    "algo": Domain.ALGORITHMS_PROGRAMMING,
    "algorithms_programming": Domain.ALGORITHMS_PROGRAMMING,
    "programming": Domain.ALGORITHMS_PROGRAMMING,
    "interview": Domain.INTERVIEW_PREP,
    "interview_prep": Domain.INTERVIEW_PREP,
    "faang": Domain.INTERVIEW_PREP,
    "ee_llm": Domain.EE_LLM_RESEARCH,
    "ee_llm_research": Domain.EE_LLM_RESEARCH,
    "phd": Domain.EE_LLM_RESEARCH,
    "llm_safety": Domain.EE_LLM_RESEARCH,
}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _display_result(result: WeighingResult, workflow_mode: WorkflowMapMode = WorkflowMapMode.STANDARD) -> None:
    """Print the WeighingResult in a human-readable format."""
    _rule()
    _print()

    if HAS_RICH:
        _print(Panel.fit(
            f"[bold cyan]{result.question}[/bold cyan]",
            title="🧪 Intuition Scientist — Results",
        ))
    else:
        _print(f"RESULTS FOR: {result.question}")

    # ---- Human intuition summary ----
    _rule("Human Intuition")
    _print(f"  Answer     : {result.human_intuition.intuitive_answer}")
    _print(f"  Confidence : {result.human_intuition.confidence:.0%}")
    if result.human_intuition.reasoning:
        _print(f"  Reasoning  : {result.human_intuition.reasoning}")

    # ---- Per-domain alignment table ----
    _rule("Domain-by-Domain Alignment")
    if HAS_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Domain", style="cyan", min_width=25)
        table.add_column("Similarity", justify="center", min_width=12)
        table.add_column("Agent confidence", justify="center", min_width=16)
        table.add_column("Key agreements", min_width=30)

        for align, resp in zip(result.alignment_scores, result.agent_responses):
            sim_str = f"{align.semantic_similarity:.2f}"
            conf_str = f"{resp.confidence:.0%}"
            agreements = ", ".join(align.key_agreements[:3]) or "—"
            table.add_row(
                align.domain.value.replace("_", " ").title(),
                sim_str,
                conf_str,
                agreements,
            )
        _console.print(table)
    else:
        for align, resp in zip(result.alignment_scores, result.agent_responses):
            _print(
                f"  {align.domain.value:<30} "
                f"similarity={align.semantic_similarity:.2f}  "
                f"agent_conf={resp.confidence:.0%}"
            )

    # ---- Agent answers ----
    _rule("Expert Agent Answers")
    for resp in result.agent_responses:
        _print(f"\n[{resp.domain.value.replace('_', ' ').title()}]")
        _print(f"  {resp.answer[:400]}")
        if resp.reasoning:
            _print(f"  Reasoning: {resp.reasoning[:200]}")

    # ---- Synthesized answer ----
    _rule("Synthesized Answer")
    _print(result.synthesized_answer)

    # ---- Overall analysis ----
    _rule("Deep Analysis")
    _print(result.overall_analysis)

    # ---- Intuition accuracy ----
    pct = result.intuition_accuracy_pct
    _rule("Intuition Accuracy Score")
    if HAS_RICH:
        colour = "green" if pct >= 70 else ("yellow" if pct >= 40 else "red")
        _print(f"  [{colour}]{pct:.1f}%[/{colour}]  (weighted alignment across all domain experts)")
    else:
        _print(f"  {pct:.1f}%  (weighted alignment across all domain experts)")

    # ---- Recommendations ----
    _rule("Recommendations")
    for i, rec in enumerate(result.recommendations, 1):
        _print(f"  {i}. {rec}")

    # ---- Workflow map (configurable verbosity) ----
    if workflow_mode is not WorkflowMapMode.OFF:
        from src.workflow import build_workflow_trace, render_workflow

        _use_mcp = any(r.mcp_context for r in result.agent_responses)
        trace = build_workflow_trace(result, use_mcp=_use_mcp)
        workflow_text = render_workflow(trace, workflow_mode)
        if workflow_text:
            _rule("Agentic Workflow")
            _print(workflow_text)

    _rule()
    _print()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intuition-scientist",
        description="Human Intuition Scientist: test your intuition against domain experts.",
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="Question to investigate (prompted interactively if omitted).",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        metavar="SPEC",
        help=(
            "Backend provider spec. Supported free/open backends: "
            "mock (default), ollama:<model>, llamacpp:<path>, "
            "groq:<model>, together:<model>, "
            "cloudflare:<model>, openrouter:<model>. "
            "Anthropic and OpenAI are not supported."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name (shorthand: equivalent to --provider <provider>:<model>). "
            "Ignored when --provider already contains a model suffix."
        ),
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        default=False,
        help="Disable internet search via MCP.",
    )
    parser.add_argument(
        "--use-mcp",
        action="store_true",
        default=False,
        help=(
            "Explicitly enable MCP internet search. "
            "Use this to override the --fast preset which disables MCP by default."
        ),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        metavar="DOMAIN",
        help=(
            "Restrict to specific domains: "
            "ee, cs, nn, social, space, physics, dl "
            "(default: auto-detect from question)."
        ),
    )
    parser.add_argument(
        "--max-domains",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of domain agents to query.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Orchestrator thread-pool size for parallel agent calls "
            "(default: 7, or 1 when --fast is set). "
            "For local Ollama on Apple Silicon, 1–2 typically gives lowest latency."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help=(
            "Enable low-latency preset optimized for local Ollama on Apple Silicon. "
            "Sets: --max-workers 1, --max-domains 3, --no-mcp, "
            "--agent-max-tokens 256, --synthesis-max-tokens 384. "
            "Any of these can be overridden by supplying the flag explicitly."
        ),
    )
    parser.add_argument(
        "--agent-max-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Token budget for each per-agent LLM call "
            "(default: 1024, or 256 when --fast is set)."
        ),
    )
    parser.add_argument(
        "--synthesis-max-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Token budget for synthesis and deep-analysis LLM calls "
            "(default: 512, or 384 when --fast is set)."
        ),
    )
    parser.add_argument(
        "--workflow-map",
        default="standard",
        choices=["off", "compact", "standard", "deep"],
        metavar="MODE",
        help=(
            "Verbosity of the agentic workflow map appended to each answer. "
            "Choices: off, compact, standard (default), deep. "
            "'deep' includes a Mermaid diagram, inputs & context, assumptions, "
            "plan, tool-call plan & results, intermediate artifacts, and next actions."
        ),
    )
    parser.add_argument(
        "--explain-workflow",
        action="store_const",
        const="deep",
        dest="workflow_map",
        help="Alias for --workflow-map deep.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Build the backend spec
    backend_spec = args.provider
    # If --model is given and --provider has no colon (i.e. is just a provider
    # name without a model suffix), combine them.
    if args.model and ":" not in backend_spec:
        backend_spec = f"{backend_spec}:{args.model}"

    # Validate: reject Anthropic/OpenAI early with a helpful error
    from src.llm.registry import get_backend as _get_backend, _BLOCKED_PROVIDERS
    provider_name = backend_spec.split(":")[0].lower()
    if provider_name in _BLOCKED_PROVIDERS:
        _print(
            f"[red]Error: {_BLOCKED_PROVIDERS[provider_name]}[/red]"
            if HAS_RICH
            else f"Error: {_BLOCKED_PROVIDERS[provider_name]}"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Apply --fast preset (each value can be overridden by explicit flag)
    # ------------------------------------------------------------------
    if args.fast:
        max_workers: int = args.max_workers if args.max_workers is not None else 1
        max_domains: Optional[int] = (
            args.max_domains if args.max_domains is not None else 3
        )
        agent_max_tokens: int = (
            args.agent_max_tokens if args.agent_max_tokens is not None else 256
        )
        synthesis_max_tokens: int = (
            args.synthesis_max_tokens if args.synthesis_max_tokens is not None else 384
        )
        # MCP: disabled by fast preset unless --use-mcp is explicitly given
        use_mcp: bool = args.use_mcp and not args.no_mcp
    else:
        max_workers = args.max_workers if args.max_workers is not None else 7
        max_domains = args.max_domains
        agent_max_tokens = (
            args.agent_max_tokens if args.agent_max_tokens is not None else 1024
        )
        synthesis_max_tokens = (
            args.synthesis_max_tokens if args.synthesis_max_tokens is not None else 512
        )
        use_mcp = not args.no_mcp

    # Parse domain overrides
    domains = None
    if args.domains:
        domains = []
        for d in args.domains:
            key = d.lower()
            if key in _DOMAIN_MAP:
                domains.append(_DOMAIN_MAP[key])
            else:
                _print(f"Unknown domain '{d}'. Choices: {', '.join(_DOMAIN_MAP)}")
                sys.exit(1)

    # Get question
    question = args.question
    if not question:
        _print()
        _print("=" * 70 if not HAS_RICH else "[bold]" + "=" * 70 + "[/bold]")
        _print("🧪  Welcome to the Human Intuition Scientist")
        _print("=" * 70 if not HAS_RICH else "[bold]" + "=" * 70 + "[/bold]")
        question = input("\nEnter your question: ").strip()
        if not question:
            _print("No question provided. Exiting.")
            sys.exit(0)

    # Import here to keep startup fast when --help is used
    from src.orchestrator.agent_orchestrator import AgentOrchestrator

    with AgentOrchestrator(
        backend_spec=backend_spec,
        use_mcp=use_mcp,
        max_workers=max_workers,
        max_domains=max_domains,
        agent_max_tokens=agent_max_tokens,
        synthesis_max_tokens=synthesis_max_tokens,
    ) as orchestrator:
        _print("\n⏳  Querying domain experts…\n" if not HAS_RICH
               else "\n[bold yellow]⏳  Querying domain experts…[/bold yellow]\n")
        result = orchestrator.run(question, domains=domains)

    workflow_mode = WorkflowMapMode(args.workflow_map)
    _display_result(result, workflow_mode=workflow_mode)


if __name__ == "__main__":
    main()
