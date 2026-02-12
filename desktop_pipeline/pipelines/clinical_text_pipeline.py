"""
Clinical text analysis pipeline.
Processes clinical notes and generates structured insights.
"""

from typing import Optional, Dict, List
import logging

from models.medgemma import MedGemmaEngine
from models.risk_model import ClinicalRiskScorer
from schemas.outputs import ClinicalTextOutput, RiskLevel
from utils.safety import get_clinical_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalTextPipeline:
    """
    End-to-end pipeline for clinical text analysis.
    Combines MedGemma LLM with risk scoring.
    """
    
    def __init__(
        self,
        use_8bit: bool = True,
        include_risk_scoring: bool = True
    ):
        """
        Initialize clinical text pipeline.
        
        Args:
            use_8bit: Use 8-bit quantization for MedGemma
            include_risk_scoring: Include risk scoring module
        """
        logger.info("Initializing Clinical Text Pipeline...")
        
        # Initialize components
        self.medgemma = MedGemmaEngine(use_8bit=use_8bit)
        
        self.risk_scorer = None
        if include_risk_scoring:
            self.risk_scorer = ClinicalRiskScorer()
        
        self.validator = get_clinical_validator()
        
        logger.info("Clinical Text Pipeline ready!")
    
    def analyze(
        self,
        clinical_note: str,
        patient_age: Optional[int] = None,
        extract_entities: bool = True,
        generate_recommendations: bool = True,
        calculate_risk: bool = True,
    ) -> ClinicalTextOutput:
        """
        Analyze clinical text and generate structured output.
        
        Args:
            clinical_note: Clinical note text
            patient_age: Patient age (optional)
            extract_entities: Extract clinical entities
            generate_recommendations: Generate recommendations
            calculate_risk: Calculate risk score
            
        Returns:
            Structured clinical text output
        """
        logger.info("Analyzing clinical text...")
        
        # Step 1: Generate clinical summary
        summary_result = self.medgemma.generate_clinical_summary(
            clinical_note,
            max_new_tokens=512,
            temperature=0.3
        )
        
        # Extract base fields
        summary = summary_result.get("summary", "")
        key_findings = summary_result.get("key_findings", [])
        
        # Step 2: Extract entities if requested
        extracted_symptoms = []
        extracted_conditions = []
        medications_mentioned = []
        
        if extract_entities:
            entities = self.medgemma.extract_clinical_entities(clinical_note)
            extracted_symptoms = entities.get("symptoms", [])
            extracted_conditions = entities.get("conditions", [])
            medications_mentioned = entities.get("medications", [])
        
        # Step 3: Calculate risk score if requested
        risk_level = RiskLevel.UNKNOWN
        confidence = 0.5
        
        if calculate_risk and self.risk_scorer:
            risk_result = self.risk_scorer.calculate_risk_score(
                symptoms=extracted_symptoms,
                conditions=extracted_conditions,
                medications=medications_mentioned,
                age=patient_age
            )
            
            risk_level = RiskLevel(risk_result["risk_level"])
            confidence = risk_result["confidence"]
        
        # Step 4: Generate recommendations if requested
        recommendations = []
        if generate_recommendations:
            recommendations = self.medgemma.generate_recommendations(
                clinical_summary=summary,
                risk_level=risk_level.value
            )
        
        # Step 5: Create structured output
        output = ClinicalTextOutput(
            summary=summary,
            key_findings=key_findings,
            extracted_symptoms=extracted_symptoms,
            extracted_conditions=extracted_conditions,
            medications_mentioned=medications_mentioned,
            risk_level=risk_level,
            confidence=confidence,
            recommendations=recommendations,
            model_version=self.medgemma.model_name
        )
        
        # Step 6: Validate and sanitize
        is_valid, issues = self.validator.validate_clinical_output(
            output.model_dump()
        )
        
        if not is_valid:
            logger.warning(f"Validation issues detected: {issues}")
            # Sanitize output
            sanitized = self.validator.sanitize_output(output.model_dump())
            output = ClinicalTextOutput(**sanitized)
        
        logger.info("Clinical text analysis complete!")
        return output
    
    def batch_analyze(
        self,
        clinical_notes: List[str],
        patient_ages: Optional[List[int]] = None,
        **kwargs
    ) -> List[ClinicalTextOutput]:
        """
        Analyze multiple clinical notes in batch.
        
        Args:
            clinical_notes: List of clinical notes
            patient_ages: Optional list of patient ages
            **kwargs: Additional arguments for analyze()
            
        Returns:
            List of clinical text outputs
        """
        logger.info(f"Batch analyzing {len(clinical_notes)} clinical notes...")
        
        results = []
        
        for i, note in enumerate(clinical_notes):
            age = patient_ages[i] if patient_ages and i < len(patient_ages) else None
            
            try:
                result = self.analyze(
                    clinical_note=note,
                    patient_age=age,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing note {i}: {e}")
                # Create error output
                error_output = ClinicalTextOutput(
                    summary=f"Error processing note: {str(e)}",
                    confidence=0.0,
                    risk_level=RiskLevel.UNKNOWN
                )
                results.append(error_output)
        
        logger.info("Batch analysis complete!")
        return results
    
    def cleanup(self):
        """Clean up pipeline and free resources"""
        if self.medgemma:
            self.medgemma.cleanup()
        
        logger.info("Clinical Text Pipeline cleaned up")


def quick_summarize(clinical_note: str, use_8bit: bool = True) -> str:
    """
    Quick helper function to summarize a clinical note.
    
    Args:
        clinical_note: Clinical note text
        use_8bit: Use 8-bit quantization
        
    Returns:
        Summary text
    """
    pipeline = ClinicalTextPipeline(use_8bit=use_8bit, include_risk_scoring=False)
    
    try:
        result = pipeline.analyze(
            clinical_note,
            extract_entities=False,
            generate_recommendations=False,
            calculate_risk=False
        )
        return result.summary
    finally:
        pipeline.cleanup()
