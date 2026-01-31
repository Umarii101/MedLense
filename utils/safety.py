"""
Safety utilities for healthcare AI outputs.
Ensures non-diagnostic language and appropriate clinical framing.
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class SafetyFramer:
    """
    Enforces safe, non-diagnostic language in medical AI outputs.
    Converts diagnostic claims to assistive observations.
    """
    
    # Diagnostic terms that should be softened
    DIAGNOSTIC_TERMS = [
        "diagnose", "diagnosed", "diagnosis",
        "definitively", "certainly", "confirmed",
        "proves", "proven", "demonstrates conclusively",
        "is", "has", "shows" # When used in diagnostic context
    ]
    
    # Safe alternative phrasings
    SAFE_ALTERNATIVES = {
        "diagnosed with": "clinical history includes",
        "diagnosis of": "assessment suggests consideration of",
        "definitively shows": "findings may indicate",
        "proves": "suggests",
        "confirmed": "reported",
        "demonstrates": "shows findings consistent with",
        "patient has": "patient presents with",
        "patient is": "clinical presentation includes",
    }
    
    # Prohibited absolute claims
    PROHIBITED_PHRASES = [
        "definitely has",
        "certainly has",
        "proven diagnosis",
        "confirmed diagnosis",
        "AI diagnosis",
        "AI-confirmed",
    ]
    
    def __init__(self):
        self.disclaimer = (
            "⚠️ ASSISTIVE TOOL ONLY - Not for diagnosis. "
            "All findings require validation by licensed healthcare provider."
        )
    
    def frame_output(self, text: str) -> str:
        """
        Convert text to safe, non-diagnostic language.
        
        Args:
            text: Raw model output
            
        Returns:
            Safety-framed text
        """
        framed_text = text
        
        # Replace diagnostic language with assistive language
        for diagnostic, safe in self.SAFE_ALTERNATIVES.items():
            framed_text = re.sub(
                diagnostic,
                safe,
                framed_text,
                flags=re.IGNORECASE
            )
        
        # Check for prohibited phrases
        violations = self._check_violations(framed_text)
        if violations:
            logger.warning(f"Safety violations detected: {violations}")
            framed_text = self._neutralize_violations(framed_text)
        
        return framed_text
    
    def _check_violations(self, text: str) -> List[str]:
        """Check for prohibited absolute claims"""
        violations = []
        text_lower = text.lower()
        
        for phrase in self.PROHIBITED_PHRASES:
            if phrase in text_lower:
                violations.append(phrase)
        
        return violations
    
    def _neutralize_violations(self, text: str) -> str:
        """Remove or soften prohibited phrases"""
        neutralized = text
        
        for phrase in self.PROHIBITED_PHRASES:
            # Replace with safer alternative
            safe_phrase = "findings suggest possible"
            neutralized = re.sub(
                phrase,
                safe_phrase,
                neutralized,
                flags=re.IGNORECASE
            )
        
        return neutralized
    
    def add_confidence_qualifier(self, text: str, confidence: float) -> str:
        """Add confidence-based qualifiers to text"""
        if confidence < 0.5:
            qualifier = "Preliminary observation suggests "
        elif confidence < 0.75:
            qualifier = "Analysis indicates "
        elif confidence < 0.9:
            qualifier = "Findings suggest "
        else:
            qualifier = "Strong indication of "
        
        # Add qualifier if not already present
        if not any(q.lower() in text.lower() for q in [
            "suggests", "indicates", "observation", "possible"
        ]):
            return f"{qualifier}{text.lower()}"
        
        return text
    
    def validate_recommendations(self, recommendations: List[str]) -> List[str]:
        """Ensure recommendations are appropriately framed"""
        validated = []
        
        for rec in recommendations:
            # Ensure recommendations are suggestions, not commands
            if rec.startswith(("You must", "Patient must", "Immediately")):
                rec = "Consider: " + rec
            
            if not rec.startswith(("Consider", "May want to", "Suggest", "Recommend")):
                rec = f"Consider {rec.lower()}"
            
            validated.append(rec)
        
        return validated
    
    def create_summary_with_disclaimers(
        self,
        summary: str,
        confidence: float,
        include_image_disclaimer: bool = False
    ) -> str:
        """Create full summary with appropriate disclaimers"""
        framed_summary = self.frame_output(summary)
        framed_summary = self.add_confidence_qualifier(framed_summary, confidence)
        
        disclaimer_text = [self.disclaimer]
        
        if confidence < 0.6:
            disclaimer_text.append(
                "⚠️ LOW CONFIDENCE - Exercise additional clinical caution."
            )
        
        if include_image_disclaimer:
            disclaimer_text.append(
                "⚠️ IMAGE ANALYSIS - Not a radiological diagnosis. "
                "Requires review by qualified radiologist."
            )
        
        full_text = f"{framed_summary}\n\n" + "\n".join(disclaimer_text)
        return full_text
    
    def check_hallucination_markers(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect potential hallucination markers in medical text.
        
        Returns:
            (has_concerns, list_of_concerns)
        """
        concerns = []
        
        # Check for overly specific claims without data
        specific_patterns = [
            r'\d+\.?\d*%',  # Specific percentages
            r'exactly \d+',  # Exact numbers
            r'precisely',
            r'specifically shows',
        ]
        
        for pattern in specific_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                concerns.append(f"Overly specific claim detected: {pattern}")
        
        # Check for missing hedging on uncertain claims
        uncertain_topics = ["prognosis", "outcome", "will", "future"]
        for topic in uncertain_topics:
            if topic in text.lower() and not any(
                hedge in text.lower() for hedge in ["may", "might", "could", "possible"]
            ):
                concerns.append(f"Missing hedge on uncertain topic: {topic}")
        
        return len(concerns) > 0, concerns


class ClinicalValidator:
    """Validates clinical outputs for safety and appropriateness"""
    
    def __init__(self):
        self.framer = SafetyFramer()
    
    def validate_clinical_output(self, output: dict) -> Tuple[bool, List[str]]:
        """
        Validate clinical output for safety concerns.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check summary
        if "summary" in output:
            violations = self.framer._check_violations(output["summary"])
            if violations:
                issues.append(f"Summary contains prohibited phrases: {violations}")
            
            has_hallucination, concerns = self.framer.check_hallucination_markers(
                output["summary"]
            )
            if has_hallucination:
                issues.extend(concerns)
        
        # Check confidence
        if "confidence" in output:
            if output["confidence"] > 0.95:
                issues.append("Confidence unrealistically high (>0.95) for assistive AI")
        
        # Ensure disclaimer present
        if "safety_disclaimer" not in output and "clinical_notes" not in output:
            issues.append("Missing required safety disclaimer")
        
        return len(issues) == 0, issues
    
    def sanitize_output(self, output: dict) -> dict:
        """Sanitize output by applying safety framing"""
        sanitized = output.copy()
        
        if "summary" in sanitized:
            sanitized["summary"] = self.framer.frame_output(sanitized["summary"])
        
        if "recommendations" in sanitized:
            sanitized["recommendations"] = self.framer.validate_recommendations(
                sanitized["recommendations"]
            )
        
        if "safety_disclaimer" not in sanitized:
            sanitized["safety_disclaimer"] = self.framer.disclaimer
        
        return sanitized


# Global instances
_safety_framer: SafetyFramer = None
_clinical_validator: ClinicalValidator = None


def get_safety_framer() -> SafetyFramer:
    """Get global safety framer instance"""
    global _safety_framer
    if _safety_framer is None:
        _safety_framer = SafetyFramer()
    return _safety_framer


def get_clinical_validator() -> ClinicalValidator:
    """Get global clinical validator instance"""
    global _clinical_validator
    if _clinical_validator is None:
        _clinical_validator = ClinicalValidator()
    return _clinical_validator
