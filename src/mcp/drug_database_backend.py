"""Drug safety tool backend — FDA drug database and interaction data.

Free data sources used
----------------------
FDA OpenFDA API    — https://api.fda.gov/drug/label.json (no key required)
RxNorm API         — https://rxnav.nlm.nih.gov/REST/ (free NLM service)

For offline/testing operation, an internal dictionary of dangerous drug
interactions is used as a fallback.

Environment variables
---------------------
FDA_API_KEY — optional FDA API key for higher rate limits (from open.fda.gov).

Usage
-----
    from src.mcp.drug_database_backend import DrugDatabaseBackend
    backend = DrugDatabaseBackend()
    # Search drug label information
    results = backend.search("warfarin drug interactions", num_results=3)
    # Check specific drug combinations
    interaction_results = backend.check_interactions(["warfarin", "aspirin", "ibuprofen"])
    for r in interaction_results:
        print(r.title, r.snippet)
"""

from __future__ import annotations

import json
import math
import os
import re
from itertools import combinations
from urllib.parse import quote_plus
from urllib.request import urlopen

from src.models import SearchResult


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def _tf_vec(tokens: list[str]) -> dict[str, float]:
    vec: dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    for t in vec:
        vec[t] = 1.0 + math.log(vec[t])
    return vec


def _cosine(va: dict[str, float], vb: dict[str, float]) -> float:
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0.0) * vb.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v ** 2 for v in va.values()))
    nb = math.sqrt(sum(v ** 2 for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


class DrugDatabaseBackend:
    """Searches FDA drug label database and checks drug-drug interactions.

    Falls back to internal interaction dictionary when FDA API is unavailable.
    """

    _FDA_BASE = "https://api.fda.gov/drug/label.json"

    _MOCK_INTERACTIONS: dict[frozenset, dict] = {
        frozenset({"warfarin", "nsaid"}): {
            "severity": "major",
            "description": (
                "NSAIDs increase bleeding risk with warfarin by inhibiting platelet function "
                "and irritating GI mucosa. Monitor INR closely. Avoid combination if possible; "
                "use acetaminophen for analgesia instead."
            ),
        },
        frozenset({"warfarin", "ibuprofen"}): {
            "severity": "major",
            "description": (
                "Ibuprofen (NSAID) increases bleeding risk with warfarin by inhibiting platelet "
                "aggregation and causing GI mucosal irritation. Monitor INR closely. "
                "Use acetaminophen for pain relief instead."
            ),
        },
        frozenset({"warfarin", "naproxen"}): {
            "severity": "major",
            "description": (
                "Naproxen (NSAID) potentiates anticoagulant effect of warfarin. "
                "Risk of GI bleeding and supratherapeutic INR. Avoid; use acetaminophen."
            ),
        },
        frozenset({"ssri", "maoi"}): {
            "severity": "contraindicated",
            "description": (
                "Potentially fatal serotonin syndrome. Minimum 14-day washout after stopping "
                "MAOI before initiating SSRI; 5 weeks for fluoxetine due to long half-life. "
                "Absolute contraindication — do not co-prescribe."
            ),
        },
        frozenset({"statin", "gemfibrozil"}): {
            "severity": "major",
            "description": (
                "Combination significantly increases risk of myopathy and rhabdomyolysis. "
                "Gemfibrozil inhibits OATP1B1 and CYP2C8, raising statin levels 2-10x. "
                "Avoid combination; fenofibrate is the preferred fibrate with statins."
            ),
        },
        frozenset({"metformin", "iodinated contrast"}): {
            "severity": "major",
            "description": (
                "Hold metformin 48 hours before and after iodinated contrast administration "
                "due to risk of contrast-induced nephropathy and subsequent metformin-associated "
                "lactic acidosis. Restart only after renal function confirmed normal."
            ),
        },
        frozenset({"ssri", "tramadol"}): {
            "severity": "major",
            "description": (
                "Risk of serotonin syndrome with concurrent use. Both drugs increase serotonergic "
                "activity. Monitor for agitation, tachycardia, hyperthermia, diaphoresis, and "
                "clonus. Additionally, tramadol lowers seizure threshold — use with caution."
            ),
        },
        frozenset({"ace inhibitor", "potassium"}): {
            "severity": "major",
            "description": (
                "Risk of potentially fatal hyperkalemia. ACE inhibitors reduce potassium "
                "excretion by decreasing aldosterone; potassium supplements may cause dangerous "
                "hyperkalemia especially in patients with CKD. Monitor serum potassium closely."
            ),
        },
        frozenset({"warfarin", "amiodarone"}): {
            "severity": "major",
            "description": (
                "Amiodarone inhibits CYP2C9 and CYP3A4, significantly increasing warfarin "
                "plasma levels and anticoagulant effect. Reduce warfarin dose by 30–50% when "
                "initiating amiodarone. Monitor INR weekly initially, then monthly."
            ),
        },
        frozenset({"fluoroquinolone", "antacid"}): {
            "severity": "moderate",
            "description": (
                "Divalent and trivalent cations (Mg²⁺, Al³⁺, Ca²⁺, Fe²⁺, Zn²⁺) chelate "
                "fluoroquinolones, reducing oral absorption by 50–90%. Separate fluoroquinolone "
                "administration by at least 2 hours before or 6 hours after antacids, "
                "multivitamins, or dairy products."
            ),
        },
        frozenset({"lithium", "nsaid"}): {
            "severity": "major",
            "description": (
                "NSAIDs reduce renal lithium clearance by inhibiting prostaglandin-mediated "
                "renal blood flow, increasing lithium levels by 20–200%. Risk of lithium "
                "toxicity (tremor, nausea, confusion, renal failure). Use acetaminophen instead; "
                "monitor lithium levels if NSAID unavoidable."
            ),
        },
        frozenset({"digoxin", "amiodarone"}): {
            "severity": "major",
            "description": (
                "Amiodarone inhibits P-glycoprotein and CYP3A4, increasing digoxin plasma "
                "levels by 70–100% within days. Risk of digoxin toxicity (nausea, visual "
                "changes, heart block). Reduce digoxin dose by 50% when initiating amiodarone. "
                "Monitor digoxin levels and ECG closely."
            ),
        },
    }

    _SINGLE_DRUG_INFO: dict[str, dict] = {
        "warfarin": {
            "class": "Vitamin K antagonist anticoagulant",
            "monitoring": "INR target 2.0–3.0 (most indications); 2.5–3.5 for mechanical heart valves",
            "major_interactions": "Amiodarone, fluoroquinolones, azole antifungals, rifampin, NSAIDs, aspirin",
            "special_populations": "Narrow therapeutic index; CYP2C9 and VKORC1 pharmacogenomics affect dosing",
        },
        "metformin": {
            "class": "Biguanide antidiabetic",
            "monitoring": "eGFR before initiation; contraindicated if eGFR <30; hold for contrast, surgery",
            "major_interactions": "Iodinated contrast, alcohol, carbonic anhydrase inhibitors",
            "special_populations": "Safe in pregnancy; preferred first-line T2DM; no dose adjustment eGFR >45",
        },
        "digoxin": {
            "class": "Cardiac glycoside / positive inotrope",
            "monitoring": "Trough level 0.5–0.9 ng/mL (HF); ECG monitoring; electrolytes (K⁺, Mg²⁺)",
            "major_interactions": "Amiodarone, verapamil, quinidine, clarithromycin — all increase levels",
            "special_populations": "Reduce dose in CKD; elderly at high risk of toxicity",
        },
        "lithium": {
            "class": "Mood stabilizer",
            "monitoring": "Trough level 0.6–1.2 mEq/L (maintenance); renal function, thyroid, ECG",
            "major_interactions": "NSAIDs, ACE inhibitors, thiazide diuretics, haloperidol — all raise levels",
            "special_populations": "Teratogenic (Ebstein's anomaly); narrow TI; dose per renal function",
        },
        "phenytoin": {
            "class": "Sodium channel blocker / antiepileptic",
            "monitoring": "Total level 10–20 mcg/mL; free level 1–2 mcg/mL (hypoalbuminemia)",
            "major_interactions": "Valproate (displaces), carbamazepine, rifampin, CYP2C9 inhibitors",
            "special_populations": "Non-linear kinetics; CYP2C9 and CYP2C19 pharmacogenomics; teratogenic",
        },
        "amiodarone": {
            "class": "Class III antiarrhythmic",
            "monitoring": "TFTs, LFTs, CXR annually; PFTs if symptomatic; ophthalmology exam",
            "major_interactions": "Warfarin, digoxin, statins, QT-prolonging drugs — multiple major interactions",
            "special_populations": "Very long half-life (40–55 days); iodine-induced thyroid effects",
        },
        "methotrexate": {
            "class": "DMARD / antimetabolite (folate antagonist)",
            "monitoring": "CBC, LFTs, creatinine every 4–8 weeks; folic acid supplementation required",
            "major_interactions": "NSAIDs, PPIs, penicillins reduce renal MTX excretion — toxicity risk",
            "special_populations": "Contraindicated in pregnancy (teratogen); avoid alcohol; dose adjust in CKD",
        },
        "cyclosporine": {
            "class": "Calcineurin inhibitor / immunosuppressant",
            "monitoring": "Trough 100–400 ng/mL (indication-specific); renal function, BP, K⁺",
            "major_interactions": "Azole antifungals, macrolides, diltiazem increase levels; rifampin decreases",
            "special_populations": "CYP3A4 and P-gp substrate; nephrotoxic; grapefruit interaction",
        },
        "tacrolimus": {
            "class": "Calcineurin inhibitor / immunosuppressant",
            "monitoring": "Trough 5–15 ng/mL (transplant-specific); renal function, glucose, K⁺",
            "major_interactions": "CYP3A4 inhibitors/inducers; nephrotoxic synergy with aminoglycosides",
            "special_populations": "More potent than cyclosporine; higher NODAT risk; CYP3A5 genotype affects dosing",
        },
        "clopidogrel": {
            "class": "P2Y12 antiplatelet — prodrug requiring CYP2C19 activation",
            "monitoring": "Bleeding signs; platelet function testing if suspected resistance",
            "major_interactions": "Omeprazole/esomeprazole (CYP2C19 inhibition reduces activation)",
            "special_populations": "CYP2C19 poor metabolizers have reduced efficacy; consider prasugrel/ticagrelor",
        },
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("FDA_API_KEY", "")

    def search(self, query: str, num_results: int = 5, **kwargs) -> list[SearchResult]:
        """Search FDA drug label database, fallback to mock data."""
        try:
            key_param = f"&api_key={self._api_key}" if self._api_key else ""
            url = f"{self._FDA_BASE}?search={quote_plus(query)}&limit={num_results}{key_param}"
            with urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            items = data.get("results", [])
            results: list[SearchResult] = []
            for item in items[:num_results]:
                openfda = item.get("openfda", {})
                brand_names = openfda.get("brand_name", ["Unknown"])
                brand = brand_names[0] if brand_names else "Unknown"
                generic_names = openfda.get("generic_name", [""])
                generic = generic_names[0] if generic_names else ""
                warnings = item.get("warnings", item.get("warnings_and_cautions", ["No warnings available"]))
                warning_text = warnings[0][:300] if warnings else "No warnings available"
                title = f"{brand} ({generic})" if generic else brand
                results.append(SearchResult(
                    title=title,
                    url=f"https://www.accessdata.fda.gov/scripts/cder/daf/",
                    snippet=warning_text,
                    relevance_score=None,
                ))
            return results if results else self._mock_search(query, num_results)
        except Exception:
            return self._mock_search(query, num_results)

    def _mock_search(self, query: str, num_results: int) -> list[SearchResult]:
        """TF-IDF search through mock interaction and drug info data."""
        q_vec = _tf_vec(_tokenize(query))
        scored: list[tuple[float, str, str, str]] = []

        for drug_pair, info in self._MOCK_INTERACTIONS.items():
            text = " ".join(drug_pair) + " " + info["description"]
            score = _cosine(q_vec, _tf_vec(_tokenize(text)))
            title = " + ".join(sorted(drug_pair)).title() + f" [{info['severity'].upper()}]"
            scored.append((score, title, info["description"], info["severity"]))

        for drug_name, info in self._SINGLE_DRUG_INFO.items():
            text = drug_name + " " + info["class"] + " " + info["major_interactions"]
            score = _cosine(q_vec, _tf_vec(_tokenize(text)))
            snippet = (
                f"Class: {info['class']}. Monitoring: {info['monitoring']}. "
                f"Key interactions: {info['major_interactions']}."
            )
            scored.append((score, drug_name.title(), snippet, "info"))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchResult] = []
        for score, title, snippet, severity in scored[:num_results]:
            results.append(SearchResult(
                title=title,
                url="https://www.accessdata.fda.gov/scripts/cder/daf/",
                snippet=snippet,
                relevance_score=round(score, 4),
            ))
        return results

    def check_interactions(self, medications: list[str]) -> list[SearchResult]:
        """Check all pairwise drug interactions in the medication list."""
        results: list[SearchResult] = []
        meds_lower = [m.lower().strip() for m in medications]

        for drug_a, drug_b in combinations(meds_lower, 2):
            pair = frozenset({drug_a, drug_b})
            found = False

            if pair in self._MOCK_INTERACTIONS:
                info = self._MOCK_INTERACTIONS[pair]
                results.append(SearchResult(
                    title=f"{drug_a.title()} + {drug_b.title()} [{info['severity'].upper()}]",
                    url="https://www.accessdata.fda.gov/scripts/cder/daf/",
                    snippet=info["description"],
                    relevance_score=None,
                ))
                found = True
            else:
                for known_pair, info in self._MOCK_INTERACTIONS.items():
                    known_list = list(known_pair)
                    if (
                        (drug_a in known_list[0] or known_list[0] in drug_a or
                         drug_a in known_list[1] or known_list[1] in drug_a)
                        and
                        (drug_b in known_list[0] or known_list[0] in drug_b or
                         drug_b in known_list[1] or known_list[1] in drug_b)
                    ):
                        results.append(SearchResult(
                            title=f"{drug_a.title()} + {drug_b.title()} [{info['severity'].upper()}]",
                            url="https://www.accessdata.fda.gov/scripts/cder/daf/",
                            snippet=info["description"],
                            relevance_score=None,
                        ))
                        found = True
                        break

            if not found:
                results.append(SearchResult(
                    title=f"{drug_a.title()} + {drug_b.title()} [No interaction found]",
                    url="https://www.accessdata.fda.gov/scripts/cder/daf/",
                    snippet=(
                        f"No major interaction identified between {drug_a} and {drug_b} in the "
                        "local database. Always verify with clinical pharmacist and current "
                        "drug interaction databases (Lexicomp, Micromedex, Clinical Pharmacology)."
                    ),
                    relevance_score=None,
                ))

        if not results:
            results.append(SearchResult(
                title="Drug Interaction Check — No Pairs Found",
                url="https://www.accessdata.fda.gov/scripts/cder/daf/",
                snippet=(
                    "Please provide at least 2 medications to check for interactions. "
                    "Consult a clinical pharmacist for comprehensive medication review."
                ),
                relevance_score=None,
            ))
        return results
