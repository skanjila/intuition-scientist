"""Tests for workflow-map / agentic-visibility feature.

Covers:
- CLI flag parsing (--workflow-map, --explain-workflow)
- WorkflowTrace construction from a WeighingResult
- render_workflow output for each mode (off, compact, standard, deep)
- Required subsections present in deep mode
- Mermaid block validity
"""

from __future__ import annotations

import re

import pytest

from src.models import (
    AgentResponse,
    AlignmentScore,
    Domain,
    HumanIntuition,
    WeighingResult,
    WorkflowMapMode,
)
from src.workflow import WORKFLOW_HEADING, build_workflow_trace, render_workflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_weighing_result(*, with_mcp: bool = False) -> WeighingResult:
    """Build a minimal WeighingResult for testing."""
    hi = HumanIntuition(
        question="How does backpropagation work?",
        intuitive_answer="Gradients flow backwards through the network.",
        confidence=0.75,
    )
    resp = AgentResponse(
        domain=Domain.NEURAL_NETWORKS,
        answer="Backpropagation computes gradients via the chain rule.",
        reasoning="Chain rule of calculus applied layer by layer.",
        confidence=0.9,
        mcp_context="web evidence about backprop" if with_mcp else "",
    )
    align = AlignmentScore(
        domain=Domain.NEURAL_NETWORKS,
        semantic_similarity=0.82,
        key_agreements=["chain rule", "gradients"],
    )
    return WeighingResult(
        question="How does backpropagation work?",
        human_intuition=hi,
        agent_responses=[resp],
        alignment_scores=[align],
        synthesized_answer="Backpropagation uses the chain rule to propagate gradients.",
        intuition_accuracy_pct=78.5,
        overall_analysis="Strong alignment between human intuition and expert analysis.",
        recommendations=["Study automatic differentiation.", "Implement a toy neural net."],
    )


# ---------------------------------------------------------------------------
# WorkflowMapMode enum
# ---------------------------------------------------------------------------


class TestWorkflowMapMode:
    def test_all_values(self) -> None:
        values = {m.value for m in WorkflowMapMode}
        assert values == {"off", "compact", "standard", "deep"}

    def test_from_string(self) -> None:
        assert WorkflowMapMode("deep") is WorkflowMapMode.DEEP
        assert WorkflowMapMode("off") is WorkflowMapMode.OFF


# ---------------------------------------------------------------------------
# build_workflow_trace
# ---------------------------------------------------------------------------


class TestBuildWorkflowTrace:
    def test_question_preserved(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        assert trace.question == result.question

    def test_steps_non_empty(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        assert len(trace.steps) >= 3

    def test_inputs_context_contains_question(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        combined = "\n".join(trace.inputs_context)
        assert result.question in combined

    def test_plan_non_empty(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        assert len(trace.plan) >= 1

    def test_assumptions_non_empty(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        assert len(trace.assumptions) >= 1

    def test_next_actions_non_empty(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        assert len(trace.next_actions) >= 1

    def test_intermediate_artifacts_non_empty(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        assert len(trace.intermediate_artifacts) >= 1

    def test_mcp_tool_call_included_when_used(self) -> None:
        result = _make_weighing_result(with_mcp=True)
        trace = build_workflow_trace(result, use_mcp=True)
        tool_names = [tc[0] for tc in trace.tool_calls]
        assert "mcp_search" in tool_names

    def test_no_tool_call_without_mcp(self) -> None:
        result = _make_weighing_result(with_mcp=False)
        trace = build_workflow_trace(result, use_mcp=False)
        assert trace.tool_calls == []

    def test_step_labels_are_strings(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        for step in trace.steps:
            assert isinstance(step.label, str) and step.label


# ---------------------------------------------------------------------------
# render_workflow — OFF mode
# ---------------------------------------------------------------------------


class TestRenderOff:
    def test_returns_empty_string(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.OFF)
        assert output == ""


# ---------------------------------------------------------------------------
# render_workflow — COMPACT mode
# ---------------------------------------------------------------------------


class TestRenderCompact:
    def test_contains_heading(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.COMPACT)
        assert WORKFLOW_HEADING in output

    def test_compact_label_in_heading(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.COMPACT)
        assert "Compact" in output

    def test_contains_mermaid_block(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.COMPACT)
        assert "```mermaid" in output and "```" in output

    def test_no_assumptions_section(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.COMPACT)
        assert "Assumptions" not in output


# ---------------------------------------------------------------------------
# render_workflow — STANDARD mode
# ---------------------------------------------------------------------------


class TestRenderStandard:
    def test_contains_heading(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.STANDARD)
        assert WORKFLOW_HEADING in output

    def test_standard_label_in_heading(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.STANDARD)
        assert "Standard" in output

    def test_contains_mermaid_block(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.STANDARD)
        assert "```mermaid" in output

    def test_contains_inputs_context(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.STANDARD)
        assert "Inputs" in output

    def test_contains_plan(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.STANDARD)
        assert "Plan" in output

    def test_no_intermediate_artifacts(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        output = render_workflow(trace, WorkflowMapMode.STANDARD)
        assert "Intermediate artifacts" not in output


# ---------------------------------------------------------------------------
# render_workflow — DEEP mode
# ---------------------------------------------------------------------------


class TestRenderDeep:
    def setup_method(self) -> None:
        result = _make_weighing_result()
        trace = build_workflow_trace(result)
        self.output = render_workflow(trace, WorkflowMapMode.DEEP)

    def test_contains_heading(self) -> None:
        assert WORKFLOW_HEADING in self.output

    def test_deep_label_in_heading(self) -> None:
        assert "Deep" in self.output

    def test_contains_mermaid_block(self) -> None:
        assert "```mermaid" in self.output

    def test_mermaid_block_has_flowchart(self) -> None:
        assert "flowchart TD" in self.output

    def test_mermaid_block_closed(self) -> None:
        # Must have an opening and at least one closing fence
        fences = [line.strip() for line in self.output.splitlines() if line.strip() == "```"]
        assert len(fences) >= 1

    def test_section_a_mermaid(self) -> None:
        assert "(A)" in self.output and "Mermaid" in self.output

    def test_section_b_inputs(self) -> None:
        assert "(B)" in self.output and "Inputs" in self.output

    def test_section_c_assumptions(self) -> None:
        assert "(C)" in self.output and "Assumptions" in self.output

    def test_section_d_plan(self) -> None:
        assert "(D)" in self.output and "Plan" in self.output

    def test_section_e_tool_calls(self) -> None:
        assert "(E)" in self.output and "Tool-call" in self.output

    def test_section_f_artifacts(self) -> None:
        assert "(F)" in self.output and "Intermediate" in self.output

    def test_section_g_next_actions(self) -> None:
        assert "(G)" in self.output and "Next actions" in self.output

    def test_contains_question_in_inputs(self) -> None:
        assert "How does backpropagation work?" in self.output

    def test_contains_alignment_table(self) -> None:
        # Markdown table header
        assert "| Domain |" in self.output

    def test_contains_acceptance_criteria(self) -> None:
        assert "Acceptance criteria" in self.output

    def test_deep_with_mcp(self) -> None:
        result = _make_weighing_result(with_mcp=True)
        trace = build_workflow_trace(result, use_mcp=True)
        output = render_workflow(trace, WorkflowMapMode.DEEP)
        assert "mcp_search" in output


# ---------------------------------------------------------------------------
# CLI flag parsing
# ---------------------------------------------------------------------------


class TestCLIFlagParsing:
    """Tests for --workflow-map and --explain-workflow flag parsing."""

    def _parse(self, argv: list[str]) -> object:
        """Return parsed args namespace (question is required but irrelevant here)."""
        import main as main_module
        parser = main_module._build_parser()
        return parser.parse_args(argv)

    def test_default_is_standard(self) -> None:
        args = self._parse(["--question", "q"])
        assert args.workflow_map == "standard"

    def test_workflow_map_deep(self) -> None:
        args = self._parse(["--question", "q", "--workflow-map", "deep"])
        assert args.workflow_map == "deep"

    def test_workflow_map_off(self) -> None:
        args = self._parse(["--question", "q", "--workflow-map", "off"])
        assert args.workflow_map == "off"

    def test_workflow_map_compact(self) -> None:
        args = self._parse(["--question", "q", "--workflow-map", "compact"])
        assert args.workflow_map == "compact"

    def test_workflow_map_standard_explicit(self) -> None:
        args = self._parse(["--question", "q", "--workflow-map", "standard"])
        assert args.workflow_map == "standard"

    def test_explain_workflow_alias_sets_deep(self) -> None:
        args = self._parse(["--question", "q", "--explain-workflow"])
        assert args.workflow_map == "deep"

    def test_invalid_mode_raises(self) -> None:
        import main as main_module
        parser = main_module._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--question", "q", "--workflow-map", "verbose"])
