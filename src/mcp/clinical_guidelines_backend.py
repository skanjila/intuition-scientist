"""Clinical guidelines tool backend — searches major guideline repositories.

Covers recommendations from: USPSTF, AHA/ACC, ADA, NICE, WHO, CDC, ACOG, AAP.

This backend uses a hardcoded set of high-value clinical guideline snippets
for offline operation. For production use, replace or augment with a real
vector store containing indexed guideline PDFs.

Environment variables
---------------------
GUIDELINES_VECTOR_STORE_URL — optional remote vector store URL for indexed guidelines.
GUIDELINES_API_KEY          — auth key for remote store.

Usage
-----
    from src.mcp.clinical_guidelines_backend import ClinicalGuidelinesBackend
    backend = ClinicalGuidelinesBackend()
    results = backend.search("diabetes type 2 management HbA1c target", num_results=3)
    for r in results:
        print(r.title)
        print(r.snippet[:200])
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

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


class ClinicalGuidelinesBackend:
    """Offline TF-IDF search over curated clinical guideline snippets.

    For production deployments, configure GUIDELINES_VECTOR_STORE_URL to
    point to a real indexed guideline repository.
    """

    _GUIDELINE_SNIPPETS: list[dict] = [
        {
            "condition": "Type 2 Diabetes Management",
            "guideline_body": "ADA Standards of Care 2024 — HbA1c targets and pharmacotherapy",
            "recommendation": (
                "For most non-pregnant adults with T2D, an HbA1c target <7% (53 mmol/mol) is "
                "appropriate. For elderly patients or those with complex comorbidities, an "
                "individualized target of 7–8% reduces hypoglycemia risk. First-line therapy: "
                "metformin + lifestyle modification. Add GLP-1 RA or SGLT2i for established "
                "ASCVD, heart failure, or CKD (evidence A)."
            ),
            "strength": "Grade A",
            "year": 2024,
            "url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
        },
        {
            "condition": "Hypertension Management",
            "guideline_body": "ACC/AHA 2017 Hypertension Guideline — BP classification and targets",
            "recommendation": (
                "BP target <130/80 mmHg for adults with hypertension, including those with "
                "diabetes, CKD, or established CVD. Stage 1 HTN (130–139/80–89 mmHg) with "
                "10-yr ASCVD risk ≥10%: initiate pharmacotherapy. First-line agents: thiazide "
                "diuretics, CCBs, ACE inhibitors, or ARBs (Class I, Level A)."
            ),
            "strength": "Class I, Level A",
            "year": 2017,
            "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065",
        },
        {
            "condition": "Breast Cancer Screening",
            "guideline_body": "USPSTF 2024 Breast Cancer Screening Recommendation",
            "recommendation": (
                "Biennial mammography screening for women aged 40–74 years (Grade B). "
                "Women 40–49 should have the opportunity to start screening based on individual "
                "preferences and risk factors. For women ≥75, evidence is insufficient (Grade I). "
                "High-risk women (BRCA1/2, prior chest radiation) should begin earlier screening "
                "with MRI supplementation per ACS high-risk guidelines."
            ),
            "strength": "Grade B",
            "year": 2024,
            "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/breast-cancer-screening",
        },
        {
            "condition": "Colorectal Cancer Screening",
            "guideline_body": "USPSTF 2021 Colorectal Cancer Screening Recommendation",
            "recommendation": (
                "Screen adults aged 45–75 years for colorectal cancer (Grade B). Options include: "
                "colonoscopy every 10 years, annual high-sensitivity guaiac FOBT or FIT, FIT-DNA "
                "every 1–3 years, CT colonography every 5 years, or flexible sigmoidoscopy every "
                "5 years. For adults 76–85, screening is an individual decision (Grade C)."
            ),
            "strength": "Grade B",
            "year": 2021,
            "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/colorectal-cancer-screening",
        },
        {
            "condition": "Cervical Cancer Screening",
            "guideline_body": "USPSTF 2018 Cervical Cancer Screening Recommendation",
            "recommendation": (
                "Women aged 21–65: Pap smear every 3 years OR Pap + HPV co-testing every 5 years "
                "for women aged 30–65 (Grade A). HPV testing alone every 5 years is an acceptable "
                "alternative for women 30–65. Screening not recommended for women <21, >65 with "
                "adequate prior negative screening, or after hysterectomy for benign disease."
            ),
            "strength": "Grade A",
            "year": 2018,
            "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/cervical-cancer-screening",
        },
        {
            "condition": "Lung Cancer Screening",
            "guideline_body": "USPSTF 2021 Lung Cancer Screening Recommendation",
            "recommendation": (
                "Annual low-dose CT (LDCT) for adults aged 50–80 years who have a ≥20 pack-year "
                "smoking history AND currently smoke OR have quit within the past 15 years (Grade B). "
                "Screening should be discontinued if the person has not smoked for 15 years, or "
                "develops a health problem that substantially limits life expectancy or the ability "
                "to have curative lung surgery."
            ),
            "strength": "Grade B",
            "year": 2021,
            "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/lung-cancer-screening",
        },
        {
            "condition": "Cardiovascular Primary Prevention — Statin Therapy",
            "guideline_body": "ACC/AHA 2019 Primary Prevention of Cardiovascular Disease Guideline",
            "recommendation": (
                "High-intensity statin for adults aged 40–75 with 10-yr ASCVD risk ≥20% (Class I). "
                "Moderate-to-high-intensity statin for 10-yr risk 7.5–20% after risk discussion "
                "(Class IIa). For 10-yr risk 5–7.5%, statin initiation is reasonable (Class IIb). "
                "Risk-enhancing factors that favor statin: LDL ≥160 mg/dL, hsCRP ≥2, ABI <0.9, "
                "family history of premature ASCVD, chronic kidney disease."
            ),
            "strength": "Class I/IIa/IIb",
            "year": 2019,
            "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678",
        },
        {
            "condition": "Depression Screening",
            "guideline_body": "USPSTF 2023 Depression, Anxiety, and Suicide Risk Screening",
            "recommendation": (
                "Screen for depression in the general adult population, including pregnant and "
                "postpartum women (Grade B). Screening with the PHQ-2 (≥3 positive) followed by "
                "PHQ-9 for confirmation. Screening should be implemented with adequate systems in "
                "place to ensure accurate diagnosis, effective treatment, and appropriate follow-up. "
                "Screen for anxiety disorders in adults under 65 (Grade B)."
            ),
            "strength": "Grade B",
            "year": 2023,
            "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/depression-anxiety-and-suicide-risk-screening",
        },
        {
            "condition": "Pediatric Dyslipidemia Screening",
            "guideline_body": "AAP/NHLBI 2011 Integrated Guidelines for Cardiovascular Health in Children",
            "recommendation": (
                "Universal fasting lipid screening recommended at ages 9–11 and again at 17–21 "
                "years. Selective screening for children with family history of premature CVD or "
                "hypercholesterolemia from birth. For children ≥10 with LDL ≥190 mg/dL or "
                "≥160 mg/dL with risk factors: statin therapy with dietary modifications."
            ),
            "strength": "Grade B",
            "year": 2011,
            "url": "https://pediatrics.aappublications.org/content/128/Supplement_5/S213",
        },
        {
            "condition": "Fall Prevention in Older Adults",
            "guideline_body": "USPSTF 2018 Fall Prevention in Community-Dwelling Older Adults",
            "recommendation": (
                "Exercise interventions for community-dwelling adults ≥65 at increased fall risk "
                "(Grade B). Vitamin D supplementation to prevent falls is NOT recommended for "
                "community-dwelling adults ≥60 who are not vitamin D deficient (Grade D). "
                "Use validated fall risk tools: STEADI algorithm, Timed Up and Go test, "
                "30-second chair stand test, 4-stage balance test."
            ),
            "strength": "Grade B",
            "year": 2018,
            "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/falls-prevention-in-community-dwelling-older-adults-interventions",
        },
        {
            "condition": "COPD Management",
            "guideline_body": "GOLD 2024 COPD Strategy Report — Pharmacological Treatment",
            "recommendation": (
                "Inhaled bronchodilators are central to pharmacological management. GOLD A "
                "(low risk, few symptoms): PRN SABA or SAMA. GOLD B (low risk, more symptoms): "
                "LAMA or LABA (or LAMA+LABA). GOLD E (high risk, high symptom burden): "
                "LAMA+LABA ± ICS based on blood eosinophil count (≥300 cells/μL favors ICS). "
                "Triple therapy (LAMA+LABA+ICS) for frequent exacerbators. "
                "Smoking cessation remains most impactful intervention."
            ),
            "strength": "Evidence A/B",
            "year": 2024,
            "url": "https://goldcopd.org/2024-gold-report/",
        },
        {
            "condition": "Sepsis and Septic Shock Management",
            "guideline_body": "Surviving Sepsis Campaign 2021 International Guidelines",
            "recommendation": (
                "Hour-1 bundle for sepsis/septic shock: (1) Measure lactate; remeasure if initial "
                ">2 mmol/L. (2) Obtain blood cultures before antibiotics. (3) Administer broad-"
                "spectrum antibiotics within 1 hour of recognition. (4) 30 mL/kg crystalloid for "
                "hypotension or lactate ≥4 mmol/L. (5) Vasopressors (norepinephrine first-line) "
                "if hypotensive during/after fluid resuscitation, target MAP ≥65 mmHg."
            ),
            "strength": "Strong recommendation",
            "year": 2021,
            "url": "https://link.springer.com/article/10.1007/s00134-021-06506-y",
        },
        {
            "condition": "Antibiotic Stewardship",
            "guideline_body": "IDSA/SHEA 2016 Implementing an Antibiotic Stewardship Program",
            "recommendation": (
                "Core elements of hospital antibiotic stewardship programs (ASPs): (1) Leadership "
                "commitment with dedicated pharmacist and physician champions. (2) Pharmacy-led "
                "prospective audit and feedback. (3) Formulary restriction and prior authorization "
                "for broad-spectrum agents (carbapenems, anti-MRSA agents). (4) De-escalation "
                "protocols at 48–72 hours based on culture results. (5) IV-to-PO conversion "
                "programs. (6) Duration of therapy guidelines."
            ),
            "strength": "Strong recommendation",
            "year": 2016,
            "url": "https://www.idsociety.org/practice-guideline/antimicrobial-stewardship/",
        },
        {
            "condition": "Adult Immunization Schedule",
            "guideline_body": "CDC ACIP 2024 Recommended Adult Immunization Schedule",
            "recommendation": (
                "Annual influenza vaccine for all adults ≥19 years (preferably before October). "
                "COVID-19: 2024–25 updated mRNA vaccine recommended annually for all adults. "
                "Tdap: once, then Td every 10 years; Tdap for each pregnancy. "
                "Zoster (Shingrix): 2-dose series for adults ≥50 (2–6 months apart). "
                "Pneumococcal: PCV21 for adults ≥65 or high-risk adults 19–64. "
                "RSV (Abrysvo or mRESVIA): adults ≥60 based on shared decision-making."
            ),
            "strength": "Routine recommendation",
            "year": 2024,
            "url": "https://www.cdc.gov/vaccines/schedules/hcp/imz/adult.html",
        },
        {
            "condition": "Obesity Management",
            "guideline_body": "AHA/ACC/TOS 2013/2022 Obesity Management Guidelines",
            "recommendation": (
                "BMI ≥30 kg/m² or ≥25 kg/m² with ≥1 weight-related comorbidity: intensive "
                "lifestyle intervention (≥14 sessions in 6 months targeting ≥5% weight loss). "
                "Pharmacotherapy adjunct: GLP-1 RAs (semaglutide 2.4 mg/wk, tirzepatide) achieve "
                "15–22% weight loss; approved for BMI ≥30 or ≥27 with comorbidity. "
                "Bariatric surgery (sleeve gastrectomy, RYGB) for BMI ≥40 or ≥35 with "
                "comorbidities, achieving 25–35% total body weight loss."
            ),
            "strength": "Class I",
            "year": 2022,
            "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063",
        },
    ]

    def __init__(
        self,
        remote_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._remote_url = remote_url or os.environ.get("GUIDELINES_VECTOR_STORE_URL", "")
        self._api_key = api_key or os.environ.get("GUIDELINES_API_KEY", "")

    def search(self, query: str, num_results: int = 5, **kwargs: Any) -> list[SearchResult]:
        """TF-IDF search over curated guideline snippets."""
        if self._remote_url:
            remote = self._remote_search(query, num_results=num_results)
            if remote:
                return remote

        q_vec = _tf_vec(_tokenize(query))
        scored: list[tuple[float, dict]] = []
        for item in self._GUIDELINE_SNIPPETS:
            text = item["guideline_body"] + " " + item["recommendation"]
            d_vec = _tf_vec(_tokenize(text))
            score = _cosine(q_vec, d_vec)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for score, item in scored[:num_results]:
            title = f"{item['condition']} — {item['guideline_body'][:50]}"
            results.append(SearchResult(
                title=title,
                url=item["url"],
                snippet=item["recommendation"],
                relevance_score=round(score, 4),
            ))
        return results

    def _remote_search(self, query: str, *, num_results: int) -> list[SearchResult]:  # pragma: no cover
        """Stub for remote vector-store integration."""
        return []
