"""PII scrubber and guardrail engine."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

MEDICAL_DISCLAIMER = (
    "AI-generated analysis only. Always requires validation by a licensed "
    "healthcare professional before any clinical decision."
)

@dataclass
class GuardrailViolation:
    rule_name: str
    severity: str
    matched_text: str
    suggestion: str = ""


class PIIScrubber:
    _PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN-REDACTED]'),
        (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b', '[CARD-REDACTED]'),
        (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE-REDACTED]'),
        (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b', '[EMAIL-REDACTED]'),
        (r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', '[IP-REDACTED]'),
        (r'\bMRN[-:\s]?\d{6,10}\b', '[MRN-REDACTED]'),
        (r'\bDOB[-:\s]?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DOB-REDACTED]'),
    ]

    def scrub(self, text: str) -> str:
        for pattern, replacement in self._PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


MEDICAL_USE_CASES = {
    "clinical_decision", "drug_interaction", "medical_literature", "patient_risk",
    "healthcare_access", "genomics_medicine", "mental_health_triage", "clinical_trials"
}


class GuardrailEngine:
    def __init__(self, rules: Optional[list[str]] = None):
        self._scrubber = PIIScrubber()
        self._rules = rules or ["pii", "prompt_injection", "medical_disclaimer", "financial_disclaimer"]

    def check_input(self, text: str) -> list[GuardrailViolation]:
        violations = []
        if "pii" in self._rules:
            if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
                violations.append(GuardrailViolation("pii_ssn", "high", "SSN detected in input", "Remove SSN before sending"))
            if re.search(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b', text):
                violations.append(GuardrailViolation("pii_email", "medium", "Email detected", "Consider anonymising"))
        if "prompt_injection" in self._rules:
            injection_patterns = ["ignore previous", "ignore all previous", "disregard your instructions", "you are now", "act as a different", "jailbreak"]
            for p in injection_patterns:
                if p.lower() in text.lower():
                    violations.append(GuardrailViolation("prompt_injection", "critical", p, "Potential prompt injection — reject input"))
                    break
        return violations

    def check_output(self, text: str, use_case: str = "") -> list[GuardrailViolation]:
        violations = []
        if use_case in MEDICAL_USE_CASES:
            disclaimer_text = "always requires validation by a licensed healthcare professional"
            if disclaimer_text.lower() not in text.lower() and "medical disclaimer" not in text.lower():
                violations.append(GuardrailViolation("missing_medical_disclaimer", "high", "Medical disclaimer absent", "Add MEDICAL_DISCLAIMER to output"))
        return violations

    def apply(self, text: str, use_case: str = "", scrub_pii: bool = True) -> tuple[str, list[GuardrailViolation]]:
        violations = self.check_input(text)
        clean = self._scrubber.scrub(text) if scrub_pii else text
        return clean, violations
