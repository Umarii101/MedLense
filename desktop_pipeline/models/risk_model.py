"""
Lightweight risk scoring model.
Demonstrates hybrid AI with interpretable numeric risk estimation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalRiskScorer:
    """
    Lightweight risk scoring model for clinical decision support.
    Uses rule-based and simple ML approaches for transparency.
    
    ⚠️ ASSISTIVE ONLY - Not for definitive risk stratification
    """
    
    # Risk factors and weights (simplified clinical rules)
    SYMPTOM_WEIGHTS = {
        "chest pain": 0.8,
        "shortness of breath": 0.7,
        "severe pain": 0.7,
        "bleeding": 0.9,
        "unconscious": 1.0,
        "seizure": 0.9,
        "confusion": 0.6,
        "fever": 0.4,
        "cough": 0.2,
        "headache": 0.3,
        "nausea": 0.3,
        "fatigue": 0.2,
    }
    
    CONDITION_WEIGHTS = {
        "diabetes": 0.3,
        "hypertension": 0.3,
        "heart disease": 0.6,
        "copd": 0.5,
        "asthma": 0.3,
        "cancer": 0.7,
        "kidney disease": 0.5,
        "liver disease": 0.5,
        "stroke": 0.8,
        "heart attack": 0.9,
    }
    
    def __init__(self):
        """Initialize risk scorer"""
        self.scaler = StandardScaler()
        self.feature_names = []
        logger.info("Clinical Risk Scorer initialized")
    
    def calculate_risk_score(
        self,
        symptoms: List[str],
        conditions: List[str],
        medications: List[str],
        age: Optional[int] = None,
        additional_factors: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate clinical risk score based on multiple factors.
        
        Args:
            symptoms: List of current symptoms
            conditions: List of medical conditions
            medications: List of current medications
            age: Patient age (if available)
            additional_factors: Additional risk factors
            
        Returns:
            Dictionary with risk score and explanation
        """
        # Initialize score components
        symptom_score = self._score_symptoms(symptoms)
        condition_score = self._score_conditions(conditions)
        medication_score = self._score_medications(medications)
        age_score = self._score_age(age) if age is not None else 0.0
        
        # Combine scores (weighted average)
        weights = {
            "symptoms": 0.4,
            "conditions": 0.3,
            "medications": 0.2,
            "age": 0.1,
        }
        
        raw_score = (
            symptom_score * weights["symptoms"] +
            condition_score * weights["conditions"] +
            medication_score * weights["medications"] +
            age_score * weights["age"]
        )
        
        # Normalize to 0-1 range
        risk_score = self._normalize_score(raw_score)
        
        # Categorize risk
        risk_level = self._categorize_risk(risk_score)
        
        # Identify contributing factors
        contributing_factors = self._identify_factors(
            symptoms, conditions, medications, age
        )
        
        # Calculate confidence (based on data completeness)
        confidence = self._calculate_confidence(
            symptoms, conditions, medications, age
        )
        
        result = {
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "component_scores": {
                "symptoms": float(symptom_score),
                "conditions": float(condition_score),
                "medications": float(medication_score),
                "age": float(age_score),
            },
            "contributing_factors": contributing_factors,
            "confidence": confidence,
            "feature_importances": self._get_feature_importances(
                symptom_score, condition_score, medication_score, age_score
            ),
            "disclaimer": "⚠️ Risk score is assistive only. Clinical judgment required.",
        }
        
        return result
    
    def _score_symptoms(self, symptoms: List[str]) -> float:
        """Score based on symptoms"""
        if not symptoms:
            return 0.0
        
        total_weight = 0.0
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for key, weight in self.SYMPTOM_WEIGHTS.items():
                if key in symptom_lower:
                    total_weight = max(total_weight, weight)
        
        return total_weight
    
    def _score_conditions(self, conditions: List[str]) -> float:
        """Score based on medical conditions"""
        if not conditions:
            return 0.0
        
        total_weight = 0.0
        matched_conditions = 0
        
        for condition in conditions:
            condition_lower = condition.lower()
            for key, weight in self.CONDITION_WEIGHTS.items():
                if key in condition_lower:
                    total_weight += weight
                    matched_conditions += 1
        
        # Average weight of matched conditions
        if matched_conditions > 0:
            return min(total_weight / max(matched_conditions, 1), 1.0)
        return 0.0
    
    def _score_medications(self, medications: List[str]) -> float:
        """Score based on medications (polypharmacy risk)"""
        if not medications:
            return 0.0
        
        # Simple polypharmacy scoring
        num_meds = len(medications)
        
        if num_meds >= 10:
            return 0.7  # High polypharmacy risk
        elif num_meds >= 5:
            return 0.4  # Moderate risk
        elif num_meds >= 3:
            return 0.2  # Low risk
        else:
            return 0.0
    
    def _score_age(self, age: int) -> float:
        """Score based on age risk"""
        if age < 18:
            return 0.1
        elif age < 40:
            return 0.0
        elif age < 60:
            return 0.2
        elif age < 75:
            return 0.4
        elif age < 85:
            return 0.6
        else:
            return 0.8
    
    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to 0-1 range"""
        # Sigmoid-like normalization
        normalized = 1 / (1 + np.exp(-5 * (raw_score - 0.5)))
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels"""
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.6:
            return "Medium"
        else:
            return "High"
    
    def _identify_factors(
        self,
        symptoms: List[str],
        conditions: List[str],
        medications: List[str],
        age: Optional[int]
    ) -> List[str]:
        """Identify contributing risk factors"""
        factors = []
        
        # High-risk symptoms
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for key, weight in self.SYMPTOM_WEIGHTS.items():
                if key in symptom_lower and weight >= 0.6:
                    factors.append(f"Concerning symptom: {symptom}")
        
        # High-risk conditions
        for condition in conditions:
            condition_lower = condition.lower()
            for key, weight in self.CONDITION_WEIGHTS.items():
                if key in condition_lower and weight >= 0.5:
                    factors.append(f"Significant condition: {condition}")
        
        # Polypharmacy
        if len(medications) >= 5:
            factors.append(f"Polypharmacy: {len(medications)} medications")
        
        # Age risk
        if age and age >= 75:
            factors.append(f"Advanced age: {age} years")
        
        return factors
    
    def _calculate_confidence(
        self,
        symptoms: List[str],
        conditions: List[str],
        medications: List[str],
        age: Optional[int]
    ) -> float:
        """Calculate confidence based on data completeness"""
        completeness_score = 0.0
        
        if symptoms:
            completeness_score += 0.3
        if conditions:
            completeness_score += 0.3
        if medications:
            completeness_score += 0.2
        if age is not None:
            completeness_score += 0.2
        
        return completeness_score
    
    def _get_feature_importances(
        self,
        symptom_score: float,
        condition_score: float,
        medication_score: float,
        age_score: float
    ) -> Dict[str, float]:
        """Get feature importance for explainability"""
        total = symptom_score + condition_score + medication_score + age_score
        
        if total == 0:
            return {
                "symptoms": 0.0,
                "conditions": 0.0,
                "medications": 0.0,
                "age": 0.0,
            }
        
        return {
            "symptoms": symptom_score / total,
            "conditions": condition_score / total,
            "medications": medication_score / total,
            "age": age_score / total,
        }
    
    def explain_risk_score(self, risk_result: Dict) -> str:
        """Generate human-readable explanation of risk score"""
        score = risk_result["risk_score"]
        level = risk_result["risk_level"]
        factors = risk_result["contributing_factors"]
        
        explanation = f"""Risk Assessment Summary:
- Overall Risk Level: {level} (score: {score:.2f})
- Confidence: {risk_result['confidence']:.0%}

Contributing Factors:
"""
        
        if factors:
            for factor in factors:
                explanation += f"  • {factor}\n"
        else:
            explanation += "  • No significant risk factors identified\n"
        
        explanation += f"""
Component Breakdown:
  • Symptoms: {risk_result['component_scores']['symptoms']:.2f}
  • Conditions: {risk_result['component_scores']['conditions']:.2f}
  • Medications: {risk_result['component_scores']['medications']:.2f}
  • Age: {risk_result['component_scores']['age']:.2f}

⚠️ This is an assistive tool only. Clinical judgment and professional assessment are required.
"""
        
        return explanation
