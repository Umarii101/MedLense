"""
Multimodal pipeline combining clinical text and image analysis.
Demonstrates integrated MedGemma + Image understanding.
"""

from PIL import Image
from typing import Optional, Dict
import logging

from models.medgemma import MedGemmaEngine
from models.image_encoder import MedicalImageEncoder
from models.risk_model import ClinicalRiskScorer
from schemas.outputs import MultimodalOutput, ClinicalTextOutput, ImageAnalysisOutput, RiskLevel
from utils.safety import get_clinical_validator
from pipelines.clinical_text_pipeline import ClinicalTextPipeline
from pipelines.image_assist_pipeline import ImageAssistPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalPipeline:
    """
    Integrated multimodal pipeline for clinical decision support.
    Combines text + image analysis with MedGemma reasoning.
    """
    
    def __init__(
        self,
        use_8bit_llm: bool = True,
        image_model_type: str = "clip"
    ):
        """
        Initialize multimodal pipeline.
        
        Args:
            use_8bit_llm: Use 8-bit quantization for LLM
            image_model_type: Image encoder type ("dino" or "clip")
        """
        logger.info("Initializing Multimodal Pipeline...")
        
        # Initialize sub-pipelines
        self.text_pipeline = ClinicalTextPipeline(
            use_8bit=use_8bit_llm,
            include_risk_scoring=True
        )
        
        self.image_pipeline = ImageAssistPipeline(
            model_type=image_model_type
        )
        
        self.validator = get_clinical_validator()
        
        logger.info("Multimodal Pipeline ready!")
    
    def analyze_clinical_text(
        self,
        clinical_note: str,
        patient_age: Optional[int] = None,
        **kwargs
    ) -> ClinicalTextOutput:
        """
        Analyze clinical text only.
        Convenience wrapper around text pipeline.
        
        Args:
            clinical_note: Clinical note text
            patient_age: Patient age (optional)
            **kwargs: Additional arguments
            
        Returns:
            Clinical text output
        """
        return self.text_pipeline.analyze(
            clinical_note=clinical_note,
            patient_age=patient_age,
            **kwargs
        )
    
    def analyze_image(
        self,
        image: Image.Image,
        image_type: str = "Unknown",
        **kwargs
    ) -> ImageAnalysisOutput:
        """
        Analyze medical image only.
        Convenience wrapper around image pipeline.
        
        Args:
            image: PIL Image
            image_type: Type of image
            **kwargs: Additional arguments
            
        Returns:
            Image analysis output
        """
        return self.image_pipeline.analyze_image(
            image=image,
            image_type=image_type,
            **kwargs
        )
    
    def analyze_with_image(
        self,
        clinical_note: str,
        image: Image.Image,
        image_type: str = "Unknown",
        patient_age: Optional[int] = None,
        integrate_findings: bool = True,
    ) -> MultimodalOutput:
        """
        Analyze clinical case with both text and image.
        Integrates findings across modalities.
        
        Args:
            clinical_note: Clinical note text
            image: Medical image
            image_type: Type of image (X-ray, CT, MRI, etc.)
            patient_age: Patient age (optional)
            integrate_findings: Use LLM to integrate findings
            
        Returns:
            Multimodal output with integrated analysis
        """
        logger.info("Performing multimodal analysis...")
        
        # Step 1: Analyze text
        logger.info("Step 1: Analyzing clinical text...")
        text_analysis = self.text_pipeline.analyze(
            clinical_note=clinical_note,
            patient_age=patient_age
        )
        
        # Step 2: Analyze image
        logger.info("Step 2: Analyzing medical image...")
        image_analysis = self.image_pipeline.analyze_image(
            image=image,
            image_type=image_type
        )
        
        # Step 3: Integrate findings
        if integrate_findings:
            logger.info("Step 3: Integrating findings...")
            integrated_analysis = self._integrate_findings(
                text_analysis=text_analysis,
                image_analysis=image_analysis,
                clinical_note=clinical_note,
                image_type=image_type
            )
        else:
            # Simple concatenation
            integrated_analysis = {
                "clinical_summary": f"{text_analysis.summary}\n\nImage: {image_type} analyzed.",
                "integrated_findings": [],
                "clinical_reasoning": "Text and image analyzed separately.",
                "next_steps": text_analysis.recommendations
            }
        
        # Step 4: Determine overall risk
        overall_risk = self._determine_overall_risk(
            text_analysis, image_analysis
        )
        
        # Step 5: Calculate overall confidence
        overall_confidence = (text_analysis.confidence + image_analysis.confidence) / 2
        
        # Step 6: Create multimodal output
        output = MultimodalOutput(
            clinical_summary=integrated_analysis["clinical_summary"],
            text_analysis=text_analysis,
            image_analysis=image_analysis,
            integrated_findings=integrated_analysis["integrated_findings"],
            overall_risk_level=overall_risk,
            overall_confidence=overall_confidence,
            clinical_reasoning=integrated_analysis["clinical_reasoning"],
            next_steps=integrated_analysis["next_steps"],
            metadata={
                "image_type": image_type,
                "patient_age": patient_age,
                "models_used": {
                    "llm": self.text_pipeline.medgemma.model_name,
                    "image": self.image_pipeline.encoder.model_type
                }
            }
        )
        
        logger.info("Multimodal analysis complete!")
        return output
    
    def analyze_from_paths(
        self,
        clinical_note: str,
        image_path: str,
        image_type: str = "Unknown",
        patient_age: Optional[int] = None,
        **kwargs
    ) -> MultimodalOutput:
        """
        Analyze from file paths.
        
        Args:
            clinical_note: Clinical note text
            image_path: Path to image file
            image_type: Type of image
            patient_age: Patient age
            **kwargs: Additional arguments
            
        Returns:
            Multimodal output
        """
        # Load image
        image = Image.open(image_path)
        
        return self.analyze_with_image(
            clinical_note=clinical_note,
            image=image,
            image_type=image_type,
            patient_age=patient_age,
            **kwargs
        )
    
    def _integrate_findings(
        self,
        text_analysis: ClinicalTextOutput,
        image_analysis: ImageAnalysisOutput,
        clinical_note: str,
        image_type: str
    ) -> Dict:
        """
        Use LLM to integrate text and image findings.
        
        Returns:
            Dictionary with integrated analysis
        """
        # Create integration prompt
        image_description = "\n".join(image_analysis.visual_observations)
        
        integration_prompt = f"""You are a medical AI assistant helping clinicians integrate multiple data sources.

Clinical Summary: {text_analysis.summary}

Key Findings from Text:
{chr(10).join(['- ' + f for f in text_analysis.key_findings])}

Image Type: {image_type}
Image Observations:
{image_description}

Task: Integrate these findings and provide:
1. A unified clinical summary
2. Findings that correlate across text and image
3. Clinical reasoning explaining the integration
4. Recommended next steps for clinical team

Provide response in JSON format with fields: clinical_summary, integrated_findings (array), clinical_reasoning, next_steps (array)

Remember: Frame as assistive observations, not diagnoses.

Response (JSON):"""
        
        # Generate integration
        try:
            integration_text = self.text_pipeline.medgemma._generate(
                integration_prompt,
                max_new_tokens=512,
                temperature=0.4
            )
            
            # Parse JSON response
            import re
            import json
            
            json_match = re.search(r'\{.*\}', integration_text, re.DOTALL)
            if json_match:
                integrated = json.loads(json_match.group())
            else:
                # Fallback
                integrated = self._fallback_integration(text_analysis, image_analysis)
        
        except Exception as e:
            logger.warning(f"Integration failed: {e}. Using fallback.")
            integrated = self._fallback_integration(text_analysis, image_analysis)
        
        return integrated
    
    def _fallback_integration(
        self,
        text_analysis: ClinicalTextOutput,
        image_analysis: ImageAnalysisOutput
    ) -> Dict:
        """Fallback integration when LLM integration fails"""
        return {
            "clinical_summary": (
                f"{text_analysis.summary} "
                f"Medical imaging shows: {', '.join(image_analysis.visual_observations[:2])}"
            ),
            "integrated_findings": [
                "Text and image findings available for review",
                f"Image quality: {image_analysis.quality_assessment}"
            ],
            "clinical_reasoning": (
                "Text and image data processed separately. "
                "Clinical correlation recommended."
            ),
            "next_steps": text_analysis.recommendations
        }
    
    def _determine_overall_risk(
        self,
        text_analysis: ClinicalTextOutput,
        image_analysis: ImageAnalysisOutput
    ) -> RiskLevel:
        """Determine overall risk level from both modalities"""
        text_risk = text_analysis.risk_level
        
        # Map risk levels to numeric scores
        risk_scores = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.UNKNOWN: 0
        }
        
        text_score = risk_scores.get(text_risk, 0)
        
        # Image quality issues can elevate risk
        image_score = 0
        if image_analysis.quality_assessment != "Good":
            image_score = 1  # Slightly elevate due to quality concerns
        
        # Take maximum risk
        max_score = max(text_score, image_score)
        
        # Map back to risk level
        if max_score >= 3:
            return RiskLevel.HIGH
        elif max_score >= 2:
            return RiskLevel.MEDIUM
        elif max_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.UNKNOWN
    
    def cleanup(self):
        """Clean up all pipelines"""
        if self.text_pipeline:
            self.text_pipeline.cleanup()
        if self.image_pipeline:
            self.image_pipeline.cleanup()
        
        logger.info("Multimodal Pipeline cleaned up")
