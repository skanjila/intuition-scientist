"""Drug Interaction agent — pharmacokinetics, drug safety, and interaction checking.

Expertise: CYP450 enzyme interactions, QT prolongation, serotonin syndrome,
high-alert medications, Beers Criteria, pregnancy/lactation safety,
therapeutic drug monitoring, narrow therapeutic index drugs.

Medical Safety Policy
---------------------
All drug interaction analysis MUST be validated by a licensed pharmacist
or physician before any prescribing or dispensing decision.

Usage
-----
    from src.agents.drug_interaction_agent import DrugInteractionAgent
    from src.llm.mock_backend import MockBackend
    agent = DrugInteractionAgent(backend=MockBackend())
    response = agent.answer("Is it safe to combine warfarin and ibuprofen?")
    print(response.answer)
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.models import Domain, SearchResult


class DrugInteractionAgent(BaseAgent):
    """Expert drug interaction and pharmacology safety agent."""

    domain = Domain.DRUG_INTERACTION

    def _build_system_prompt(self) -> str:
        return (
            "You are a clinical pharmacist and pharmacologist with expertise in drug safety, "
            "pharmacokinetics, pharmacodynamics, and drug-drug interactions.\n\n"
            "=== PHARMACOLOGY EXPERTISE ===\n"
            "- Pharmacokinetics (ADME): absorption, distribution, metabolism, excretion\n"
            "- Pharmacodynamics: receptor mechanisms, dose-response relationships\n"
            "- CYP450 enzyme interactions:\n"
            "  * CYP2D6: codeine, tramadol, tamoxifen, TCAs, metoprolol, haloperidol\n"
            "  * CYP2C9: warfarin, NSAIDs, phenytoin, sulfonylureas\n"
            "  * CYP2C19: clopidogrel, PPIs, SSRIs, diazepam\n"
            "  * CYP3A4/5: statins, calcium channel blockers, macrolides, azole antifungals,\n"
            "    immunosuppressants (cyclosporine, tacrolimus), HIV protease inhibitors\n"
            "- Protein binding displacement: warfarin, phenytoin, valproate\n"
            "- QT prolongation risk: QTc calculation (Bazett/Fridericia), torsades de pointes risk,\n"
            "  high-risk drugs (antipsychotics, antiarrhythmics, fluoroquinolones, azithromycin)\n"
            "- Serotonin syndrome: diagnosis (Hunter criteria), implicated drugs (SSRIs, SNRIs,\n"
            "  MAOIs, tramadol, linezolid, triptans, lithium, fentanyl), management\n"
            "- Anticholinergic burden: Anticholinergic Cognitive Burden (ACB) scale, cumulative\n"
            "  burden in elderly, associated cognitive impairment and fall risk\n"
            "- High-alert medications: anticoagulants (warfarin, DOACs, heparin), insulin,\n"
            "  opioids, digoxin, lithium, phenytoin, methotrexate, concentrated electrolytes\n"
            "- Beers Criteria (AGS 2023): inappropriate medications in elderly (≥65 years),\n"
            "  PIMs (potentially inappropriate medications), drug-disease interactions\n"
            "- Pregnancy/lactation safety: FDA categories A/B/C/D/X, TERIS risk ratings,\n"
            "  LactMed database, compatible vs. contraindicated drugs in breastfeeding\n"
            "- Pediatric dosing: weight-based calculations (mg/kg), BSA-based dosing,\n"
            "  off-label use, pediatric formulation considerations\n"
            "- Renal dose adjustment: Cockcroft-Gault GFR estimation, MDRD, CKD-EPI,\n"
            "  dose reductions required for renally cleared drugs\n"
            "- Hepatic dose adjustment: Child-Pugh classification (A/B/C),\n"
            "  MELD score, hepatically metabolized drugs requiring adjustment\n"
            "- Therapeutic drug monitoring: trough level sampling times,\n"
            "  target ranges (vancomycin AUC/MIC, aminoglycosides, digoxin, lithium,\n"
            "  phenytoin, cyclosporine, tacrolimus, valproate, carbamazepine)\n"
            "- Generic vs. brand bioequivalence: 80–125% rule, NTI drugs requiring brand\n"
            "- Narrow therapeutic index (NTI) drugs: warfarin, digoxin, lithium, phenytoin,\n"
            "  cyclosporine, tacrolimus, theophylline, levothyroxine, carbamazepine\n\n"
            "=== RESPONSE FORMAT ===\n"
            "For drug interaction queries, provide:\n"
            "1. Interaction severity (contraindicated / major / moderate / minor / none)\n"
            "2. Mechanism of interaction (PK vs. PD)\n"
            "3. Clinical consequences and risk factors\n"
            "4. Management recommendations (avoid, monitor, dose adjust, alternative)\n"
            "5. Monitoring parameters if combination continued\n"
            "6. Special population considerations (elderly, renal/hepatic impairment)\n\n"
            "CRITICAL: Always include the medical disclaimer in your response.\n"
            "Respond only with the requested JSON structure."
        )

    def _compute_weights(
        self, question: str, mcp_results: list[SearchResult]
    ) -> tuple[float, float]:
        """Drug interaction data is highly fact-based — boost tool weight."""
        mcp_quality = min(1.0, len(mcp_results) / 4)
        tool_w = min(0.80, 0.65 + mcp_quality * 0.15)
        return round(1.0 - tool_w, 3), round(tool_w, 3)
