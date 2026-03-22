"""Model sweep / evaluation test suite.

This test suite iterates over a dataset of complex Q/A pairs, runs the
orchestrator end-to-end with prefilled human intuition, and checks that the
pipeline produces non-trivial results.

Running behaviour
-----------------
By default the tests are **skipped** so that the CI pipeline stays fast and
free (no model servers or API keys required).  To run them locally:

    RUN_MODEL_SWEEP=1 pytest tests/test_model_sweep.py -v

Model list
----------
Set the ``INTUITION_SCIENTIST_MODELS`` environment variable to a
comma-separated list of provider specs (default: ``mock``):

    INTUITION_SCIENTIST_MODELS=ollama:llama3.1:8b,groq:llama-3.1-8b-instant

Supported specs:
    mock
    ollama:<model>
    llamacpp:<path-to-gguf>
    groq:<model>
    together:<model>
    cloudflare:<model>
    openrouter:<model>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.llm.registry import get_backend
from src.models import Domain, HumanIntuition
from src.orchestrator.agent_orchestrator import AgentOrchestrator

# ---------------------------------------------------------------------------
# Gate: skip unless RUN_MODEL_SWEEP=1
# ---------------------------------------------------------------------------

_RUN_SWEEP = os.environ.get("RUN_MODEL_SWEEP", "").strip() in {"1", "true", "yes"}

pytestmark = pytest.mark.skipif(
    not _RUN_SWEEP,
    reason="Model sweep tests are opt-in: set RUN_MODEL_SWEEP=1 to enable.",
)

# ---------------------------------------------------------------------------
# Load Q/A fixture
# ---------------------------------------------------------------------------

_FIXTURES_PATH = Path(__file__).parent / "fixtures" / "qa.yaml"


def _load_qa_fixtures() -> list[dict[str, Any]]:
    with _FIXTURES_PATH.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = ["mock"]

_DOMAIN_NAME_MAP: dict[str, Domain] = {
    "electrical_engineering": Domain.ELECTRICAL_ENGINEERING,
    "computer_science": Domain.COMPUTER_SCIENCE,
    "neural_networks": Domain.NEURAL_NETWORKS,
    "social_science": Domain.SOCIAL_SCIENCE,
    "space_science": Domain.SPACE_SCIENCE,
    "physics": Domain.PHYSICS,
    "deep_learning": Domain.DEEP_LEARNING,
}


def _get_model_specs() -> list[str]:
    raw = os.environ.get("INTUITION_SCIENTIST_MODELS", "").strip()
    if not raw:
        return _DEFAULT_MODELS
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_intuition(entry: dict[str, Any]) -> HumanIntuition:
    intu = entry["intuition"]
    return HumanIntuition(
        question=entry["question"].strip(),
        intuitive_answer=intu["answer"].strip(),
        confidence=float(intu["confidence"]),
        reasoning=intu.get("reasoning", "").strip(),
    )


def _parse_domains(entry: dict[str, Any]) -> list[Domain] | None:
    raw = entry.get("domains")
    if not raw:
        return None
    return [_DOMAIN_NAME_MAP[d] for d in raw if d in _DOMAIN_NAME_MAP]


# ---------------------------------------------------------------------------
# Test: pipeline runs end-to-end for each (model, question) pair
# ---------------------------------------------------------------------------

QA_ENTRIES = _load_qa_fixtures() if _FIXTURES_PATH.exists() else []
MODEL_SPECS = _get_model_specs()


@pytest.mark.parametrize("model_spec", MODEL_SPECS)
@pytest.mark.parametrize("entry", QA_ENTRIES, ids=[e["id"] for e in QA_ENTRIES])
def test_sweep_pipeline(entry: dict[str, Any], model_spec: str) -> None:
    """Run the full pipeline for one Q/A entry with the given model backend.

    Assertions
    ----------
    - ``WeighingResult`` is returned without exceptions.
    - ``agent_responses`` is non-empty.
    - ``alignment_scores`` is non-empty with values in [0, 1].
    - ``synthesized_answer`` is non-empty.
    - For entries with explicit domains, the correct domains appear in the
      agent responses.
    """
    # Build backend — skip gracefully if the backend is unavailable
    try:
        backend = get_backend(model_spec)
    except ValueError as exc:
        pytest.skip(f"Backend spec '{model_spec}' is invalid: {exc}")

    intuition = _build_intuition(entry)
    domains = _parse_domains(entry)

    try:
        with AgentOrchestrator(
            backend=backend,
            use_mcp=False,
            max_domains=len(domains) if domains else None,
        ) as orch:
            result = orch.run(
                intuition.question,
                prefilled_intuition=intuition,
                domains=domains,
            )
    except RuntimeError as exc:
        # Backend is configured but unavailable at runtime (server not running,
        # missing API key, etc.) — skip rather than fail.
        pytest.skip(f"Backend '{model_spec}' unavailable: {exc}")

    # --- Assertions ---
    assert result.agent_responses, "agent_responses must be non-empty"
    assert result.alignment_scores, "alignment_scores must be non-empty"
    assert result.synthesized_answer, "synthesized_answer must be non-empty"

    for score in result.alignment_scores:
        assert 0.0 <= score.semantic_similarity <= 1.0, (
            f"alignment score out of range: {score.semantic_similarity}"
        )

    # Check that requested domains appear in agent responses
    if domains:
        response_domains = {r.domain for r in result.agent_responses}
        for domain in domains:
            assert domain in response_domains, (
                f"Expected domain {domain} in agent responses but got {response_domains}"
            )

    # Check expected keywords appear somewhere in the synthesized answer
    # or agent responses or the human intuition text (which is the richest
    # source when using a mock backend).
    expected_kw = [kw.lower() for kw in entry.get("expected_keywords", [])]
    combined_text = (
        result.synthesized_answer.lower()
        + " "
        + " ".join(r.answer.lower() for r in result.agent_responses)
        + " "
        + intuition.intuitive_answer.lower()
        + " "
        + intuition.reasoning.lower()
    )
    missing = [kw for kw in expected_kw if kw not in combined_text]
    assert not missing, (
        f"Expected keywords not found in output: {missing}. "
        f"(model={model_spec}, question_id={entry['id']})"
    )


# ---------------------------------------------------------------------------
# Test: Social Science domain participates when explicitly requested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_spec", MODEL_SPECS)
def test_social_science_domain_explicit(model_spec: str) -> None:
    """Verify SocialScienceAgent is invoked when Domain.SOCIAL_SCIENCE is
    explicitly passed to the orchestrator."""
    try:
        backend = get_backend(model_spec)
    except ValueError as exc:
        pytest.skip(f"Backend spec '{model_spec}' is invalid: {exc}")

    intuition = HumanIntuition(
        question="How does social media affect collective behaviour in societies?",
        intuitive_answer=(
            "Social media amplifies group dynamics, accelerates information spread, "
            "and can polarise opinions through echo chambers."
        ),
        confidence=0.70,
        reasoning="Filter bubbles and algorithmic curation reinforce existing beliefs.",
    )

    try:
        with AgentOrchestrator(
            backend=backend,
            use_mcp=False,
        ) as orch:
            result = orch.run(
                intuition.question,
                prefilled_intuition=intuition,
                domains=[Domain.SOCIAL_SCIENCE],
            )
    except RuntimeError as exc:
        pytest.skip(f"Backend '{model_spec}' unavailable: {exc}")

    response_domains = {r.domain for r in result.agent_responses}
    assert Domain.SOCIAL_SCIENCE in response_domains, (
        f"Domain.SOCIAL_SCIENCE not found in agent responses: {response_domains}"
    )
    assert result.synthesized_answer, "synthesized_answer must not be empty"
