"""Tests for open-source model registry."""
from __future__ import annotations
import pytest
from src.llm.model_registry import (
    MODEL_CATALOG, USE_CASE_MODEL_RECOMMENDATIONS, FREE_FALLBACK_CHAIN,
    get_model_for_use_case, get_backend_with_fallback, list_free_models, setup_instructions,
)
from src.llm.mock_backend import MockBackend


class TestModelCatalog:
    def test_has_minimum_models(self):
        assert len(MODEL_CATALOG) >= 15

    def test_all_specs_have_provider(self):
        for key in MODEL_CATALOG:
            assert ":" in key or key == "mock"

    def test_fallback_chain_ends_mock(self):
        assert FREE_FALLBACK_CHAIN[-1] == "mock"


class TestUseCaseRecommendations:
    def test_all_use_cases_have_recommendations(self):
        expected = [
            "customer_support","incident_response","finance_reconciliation",
            "code_review","analytics","rfp_response","compliance_qa",
            "sales_outreach","supply_chain_exception",
            "clinical_decision_support","drug_interaction","medical_literature",
            "patient_risk","healthcare_access","genomics_medicine",
            "mental_health_triage","clinical_trials",
        ]
        for uc in expected:
            assert uc in USE_CASE_MODEL_RECOMMENDATIONS, f"{uc} missing"

    def test_each_use_case_has_profiles(self):
        for uc, profiles in USE_CASE_MODEL_RECOMMENDATIONS.items():
            assert "fast" in profiles or "balanced" in profiles, f"{uc} missing fast/balanced"


class TestGetModel:
    def test_returns_string(self):
        result = get_model_for_use_case("customer_support", "balanced")
        assert isinstance(result, str)

    def test_fallback_to_balanced(self):
        result = get_model_for_use_case("customer_support", "nonexistent_profile")
        assert isinstance(result, str)

    def test_unknown_use_case_returns_fallback(self):
        result = get_model_for_use_case("unknown_use_case", "balanced")
        assert isinstance(result, str)


class TestGetBackendWithFallback:
    def test_returns_backend(self):
        backend = get_backend_with_fallback("customer_support", "balanced")
        assert backend is not None

    def test_returns_mock_as_fallback(self):
        backend = get_backend_with_fallback("unknown", "balanced")
        assert backend is not None


class TestListFreeModels:
    def test_returns_list(self):
        models = list_free_models()
        assert isinstance(models, list)
        assert len(models) >= 1


class TestSetupInstructions:
    def test_returns_string(self):
        inst = setup_instructions("customer_support", "balanced")
        assert isinstance(inst, str)
        assert len(inst) > 0
