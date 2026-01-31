"""
Pydantic schemas for structured medical AI outputs.
Ensures type safety and validation for all system outputs.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class RiskLevel(str, Enum):
    """Risk stratification levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


class ConfidenceLevel(str, Enum):
    """Confidence indicators"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class ClinicalTextOutput(BaseModel):
    """Output schema for clinical text analysis"""
    
    summary: str = Field(
        description="Brief clinical summary in non-diagnostic language"
    )
    
    key_findings: List[str] = Field(
        default_factory=list,
        description="Important observations from clinical note"
    )
    
    extracted_symptoms: List[str] = Field(
        default_factory=list,
        description="Symptoms mentioned in text"
    )
    
    extracted_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions or diagnoses mentioned (historical)"
    )
    
    medications_mentioned: List[str] = Field(
        default_factory=list,
        description="Medications referenced in note"
    )
    
    risk_level: RiskLevel = Field(
        default=RiskLevel.UNKNOWN,
        description="Estimated risk level for clinical review priority"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score (0-1)"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="Suggested next steps for clinician consideration"
    )
    
    safety_disclaimer: str = Field(
        default="⚠️ ASSISTIVE TOOL ONLY - Not for diagnosis. Requires validation by licensed healthcare provider.",
        description="Mandatory safety disclaimer"
    )
    
    model_version: str = Field(
        default="medgemma-7b",
        description="Model used for inference"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in valid range"""
        return max(0.0, min(1.0, v))
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level"""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class ImageAnalysisOutput(BaseModel):
    """Output schema for medical image analysis"""
    
    image_type: str = Field(
        default="Unknown",
        description="Type of medical image (X-ray, CT, MRI, etc.)"
    )
    
    visual_observations: List[str] = Field(
        default_factory=list,
        description="Observable features (assistive, not diagnostic)"
    )
    
    embedding_shape: Optional[tuple] = Field(
        default=None,
        description="Shape of image embedding vector"
    )
    
    quality_assessment: str = Field(
        default="Unable to assess",
        description="Image quality notes"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in feature extraction"
    )
    
    safety_disclaimer: str = Field(
        default="⚠️ IMAGE ANALYSIS IS ASSISTIVE ONLY - Not a radiological diagnosis. Must be reviewed by qualified radiologist.",
        description="Image-specific safety disclaimer"
    )


class MultimodalOutput(BaseModel):
    """Output schema for combined text + image analysis"""
    
    clinical_summary: str = Field(
        description="Integrated summary combining text and image analysis"
    )
    
    text_analysis: ClinicalTextOutput = Field(
        description="Results from clinical text processing"
    )
    
    image_analysis: Optional[ImageAnalysisOutput] = Field(
        default=None,
        description="Results from image processing (if image provided)"
    )
    
    integrated_findings: List[str] = Field(
        default_factory=list,
        description="Findings that correlate across modalities"
    )
    
    overall_risk_level: RiskLevel = Field(
        default=RiskLevel.UNKNOWN,
        description="Combined risk assessment"
    )
    
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall system confidence"
    )
    
    clinical_reasoning: str = Field(
        default="",
        description="Explanation of how findings were integrated"
    )
    
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended actions for clinical team"
    )
    
    safety_disclaimer: str = Field(
        default="⚠️ MULTIMODAL ANALYSIS IS ASSISTIVE ONLY - All findings require clinical validation. Not a substitute for professional medical judgment.",
        description="Comprehensive safety disclaimer"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional system metadata"
    )


class RiskScoreOutput(BaseModel):
    """Output schema for risk scoring models"""
    
    risk_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Numerical risk score (0-1)"
    )
    
    risk_level: RiskLevel = Field(
        description="Categorical risk level"
    )
    
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to risk score"
    )
    
    feature_importances: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance scores for explainability"
    )
    
    confidence_interval: Optional[tuple] = Field(
        default=None,
        description="95% confidence interval for risk score"
    )
    
    model_name: str = Field(
        default="baseline-risk-model",
        description="Risk model identifier"
    )


class SystemHealthCheck(BaseModel):
    """System status and health monitoring"""
    
    gpu_available: bool = Field(description="CUDA GPU availability")
    gpu_name: Optional[str] = Field(default=None, description="GPU model name")
    gpu_memory_allocated: Optional[float] = Field(default=None, description="GPU memory in GB")
    gpu_memory_reserved: Optional[float] = Field(default=None, description="Reserved GPU memory in GB")
    
    models_loaded: List[str] = Field(
        default_factory=list,
        description="Successfully loaded models"
    )
    
    system_ready: bool = Field(description="Overall system readiness")
    
    warnings: List[str] = Field(
        default_factory=list,
        description="System warnings or issues"
    )
