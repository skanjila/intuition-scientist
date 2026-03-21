"""Tests for all domain agents (offline/mock mode — no LLM key required)."""

import pytest
from src.agents.base_agent import BaseAgent
from src.agents.computer_science_agent import ComputerScienceAgent
from src.agents.deep_learning_agent import DeepLearningAgent
from src.agents.electrical_engineering_agent import ElectricalEngineeringAgent
from src.agents.neural_networks_agent import NeuralNetworksAgent
from src.agents.physics_agent import PhysicsAgent
from src.agents.social_science_agent import SocialScienceAgent
from src.agents.space_science_agent import SpaceScienceAgent
# High-economic-value industry agents
from src.agents.healthcare_agent import HealthcareAgent
from src.agents.climate_energy_agent import ClimateEnergyAgent
from src.agents.finance_economics_agent import FinanceEconomicsAgent
from src.agents.cybersecurity_agent import CybersecurityAgent
from src.agents.biotech_genomics_agent import BiotechGenomicsAgent
from src.agents.supply_chain_agent import SupplyChainAgent
from src.models import AgentResponse, Domain


ALL_AGENT_CLASSES = [
    ElectricalEngineeringAgent,
    ComputerScienceAgent,
    NeuralNetworksAgent,
    SocialScienceAgent,
    SpaceScienceAgent,
    PhysicsAgent,
    DeepLearningAgent,
    HealthcareAgent,
    ClimateEnergyAgent,
    FinanceEconomicsAgent,
    CybersecurityAgent,
    BiotechGenomicsAgent,
    SupplyChainAgent,
]

EXPECTED_DOMAINS = [
    Domain.ELECTRICAL_ENGINEERING,
    Domain.COMPUTER_SCIENCE,
    Domain.NEURAL_NETWORKS,
    Domain.SOCIAL_SCIENCE,
    Domain.SPACE_SCIENCE,
    Domain.PHYSICS,
    Domain.DEEP_LEARNING,
    Domain.HEALTHCARE,
    Domain.CLIMATE_ENERGY,
    Domain.FINANCE_ECONOMICS,
    Domain.CYBERSECURITY,
    Domain.BIOTECH_GENOMICS,
    Domain.SUPPLY_CHAIN,
]


class TestAgentDomains:
    def test_all_agents_have_correct_domain(self):
        for cls, expected in zip(ALL_AGENT_CLASSES, EXPECTED_DOMAINS):
            assert cls.domain == expected, f"{cls.__name__} has wrong domain"


class TestBaseAgentMockMode:
    """Run agents without an LLM key — exercises mock fallback path."""

    def _make_agent(self, cls):
        # Pass llm_provider with a deliberately blank key → falls back to mock
        agent = cls(mcp_client=None, llm_provider="anthropic")
        # Force the client to None so mock path is used regardless of env
        agent._llm_client = None
        return agent

    @pytest.mark.parametrize("cls", ALL_AGENT_CLASSES)
    def test_answer_returns_agent_response(self, cls):
        agent = self._make_agent(cls)
        resp = agent.answer("What is energy?")
        assert isinstance(resp, AgentResponse)
        assert resp.domain == cls.domain
        assert isinstance(resp.answer, str)
        assert len(resp.answer) > 0
        assert 0.0 <= resp.confidence <= 1.0

    @pytest.mark.parametrize("cls", ALL_AGENT_CLASSES)
    def test_answer_sources_is_list(self, cls):
        agent = self._make_agent(cls)
        resp = agent.answer("Explain recursion")
        assert isinstance(resp.sources, list)

    def test_system_prompt_not_empty(self):
        for cls in ALL_AGENT_CLASSES:
            agent = cls(mcp_client=None)
            prompt = agent._build_system_prompt()
            assert isinstance(prompt, str)
            assert len(prompt) > 50, f"{cls.__name__} system prompt is too short"


class TestBaseAgentJsonParsing:
    """Test the JSON extraction helper."""

    def test_extract_valid_json(self):
        text = 'some text {"answer": "42", "confidence": 0.9} more text'
        result = BaseAgent._extract_json(text)
        assert result == {"answer": "42", "confidence": 0.9}

    def test_extract_invalid_returns_none(self):
        assert BaseAgent._extract_json("no json here") is None

    def test_extract_malformed_returns_none(self):
        assert BaseAgent._extract_json("{bad json") is None
