"""
Medical image analysis pipeline.
Provides assistive visual feature extraction (NOT diagnostic).
"""

from PIL import Image
from typing import Optional, Dict
import logging

from models.image_encoder import MedicalImageEncoder, create_medical_image_description
from schemas.outputs import ImageAnalysisOutput
from utils.safety import get_safety_framer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAssistPipeline:
    """
    Pipeline for assistive medical image analysis.
    
    ⚠️ NOT FOR RADIOLOGICAL DIAGNOSIS
    Provides visual feature extraction to support clinical review.
    """
    
    def __init__(
        self,
        model_type: str = "clip",  # "dino" or "clip"
        image_size: int = 224
    ):
        """
        Initialize image assist pipeline.
        
        Args:
            model_type: Image encoder type ("dino" or "clip")
            image_size: Input image size
        """
        logger.info("Initializing Image Assist Pipeline...")
        
        self.encoder = MedicalImageEncoder(
            model_type=model_type,
            image_size=image_size
        )
        
        self.safety_framer = get_safety_framer()
        
        logger.info("Image Assist Pipeline ready!")
    
    def analyze_image(
        self,
        image: Image.Image,
        image_type: str = "Unknown",
        include_quality_check: bool = True,
        text_prompts: Optional[list] = None
    ) -> ImageAnalysisOutput:
        """
        Analyze medical image and extract features.
        
        Args:
            image: PIL Image
            image_type: Type of image (X-ray, CT, MRI, etc.)
            include_quality_check: Perform image quality assessment
            text_prompts: Optional text prompts for guided analysis (CLIP only)
            
        Returns:
            Image analysis output
        """
        logger.info(f"Analyzing {image_type} image...")
        
        # Step 1: Quality assessment
        quality_assessment = "Not assessed"
        quality_metrics = {
            "size": image.size,
            "mode": image.mode,
            "mean_intensity": None,
            "std_intensity": None,
            "dynamic_range": None,
            "quality_issues": [],
            "quality_assessment": "Not assessed"
        }
        if include_quality_check:
            quality_metrics = self.encoder.assess_image_quality(image)
            quality_assessment = quality_metrics["quality_assessment"]
            
            if quality_metrics["quality_issues"]:
                logger.warning(f"Image quality issues: {quality_metrics['quality_issues']}")
        
        # Step 2: Extract image features
        if text_prompts:
            # Text-guided feature extraction (CLIP only)
            features = self.encoder.get_visual_features_for_text(
                image,
                text_prompts=text_prompts
            )
        else:
            # Standard feature extraction
            features = self.encoder.encode_image(image)
        
        # Step 3: Generate visual observations
        visual_observations = self._generate_observations(features, quality_metrics)
        
        # Step 4: Determine confidence
        confidence = self._calculate_confidence(features, quality_metrics)
        
        # Step 5: Create output
        output = ImageAnalysisOutput(
            image_type=image_type,
            visual_observations=visual_observations,
            embedding_shape=tuple(features["embedding_shape"]),
            quality_assessment=quality_assessment,
            confidence=confidence
        )
        
        logger.info("Image analysis complete!")
        return output
    
    def analyze_image_from_path(
        self,
        image_path: str,
        image_type: str = "Unknown",
        **kwargs
    ) -> ImageAnalysisOutput:
        """
        Analyze image from file path.
        
        Args:
            image_path: Path to image file
            image_type: Type of image
            **kwargs: Additional arguments for analyze_image()
            
        Returns:
            Image analysis output
        """
        try:
            image = Image.open(image_path)
            return self.analyze_image(image, image_type=image_type, **kwargs)
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            raise
    
    def get_image_features_for_llm(
        self,
        image: Image.Image,
        image_type: str = "Unknown"
    ) -> str:
        """
        Get image features formatted for LLM context.
        Useful for multimodal analysis.
        
        Args:
            image: PIL Image
            image_type: Type of image
            
        Returns:
            Formatted text description
        """
        # Extract features
        features = self.encoder.encode_image(image)
        quality_metrics = self.encoder.assess_image_quality(image)
        
        # Create description
        description = create_medical_image_description(features, quality_metrics)
        
        # Add image type
        full_description = f"Image Type: {image_type}\n{description}"
        
        return full_description
    
    def _generate_observations(
        self,
        features: Dict,
        quality_metrics: Dict
    ) -> list:
        """Generate visual observations from features"""
        observations = []
        
        # Image properties
        observations.append(
            f"Image dimensions: {quality_metrics['size']}"
        )
        
        # Quality notes
        if quality_metrics["quality_issues"]:
            for issue in quality_metrics["quality_issues"]:
                observations.append(f"Quality note: {issue}")
        
        # Feature extraction success
        observations.append(
            f"Features extracted using {features['model_type']} model "
            f"({features['embedding_dim']} dimensions)"
        )
        
        # Text similarity observations (if available)
        if "text_similarities" in features:
            high_sim = [
                (prompt, sim)
                for prompt, sim in features["text_similarities"].items()
                if sim > 0.4
            ]
            if high_sim:
                observations.append(
                    "Visual characteristics detected: " +
                    ", ".join([f"{p} ({s:.0%})" for p, s in high_sim])
                )
        
        return observations
    
    def _calculate_confidence(
        self,
        features: Dict,
        quality_metrics: Dict
    ) -> float:
        """Calculate confidence score for image analysis"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for good quality
        if quality_metrics["quality_assessment"] == "Good":
            confidence += 0.2
        
        # Decrease for quality issues
        if quality_metrics["quality_issues"]:
            confidence -= 0.1 * min(len(quality_metrics["quality_issues"]), 3)
        
        # Increase for successful feature extraction
        if features["embedding"] is not None:
            confidence += 0.1
        
        # Clamp to valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def compare_images(
        self,
        image1: Image.Image,
        image2: Image.Image,
        image1_type: str = "Baseline",
        image2_type: str = "Follow-up"
    ) -> Dict:
        """
        Compare two medical images (e.g., baseline vs follow-up).
        
        Args:
            image1: First image
            image2: Second image
            image1_type: Description of first image
            image2_type: Description of second image
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {image1_type} vs {image2_type}...")
        
        # Extract features from both images
        features1 = self.encoder.encode_image(image1)
        features2 = self.encoder.encode_image(image2)
        
        # Calculate similarity
        import numpy as np
        emb1 = features1["embedding"].flatten()
        emb2 = features2["embedding"].flatten()
        
        # Cosine similarity
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
        # Interpret similarity
        if similarity > 0.95:
            interpretation = "Very similar - minimal change detected"
        elif similarity > 0.85:
            interpretation = "Similar - minor changes may be present"
        elif similarity > 0.70:
            interpretation = "Moderate similarity - notable changes detected"
        else:
            interpretation = "Significant differences detected"
        
        result = {
            "similarity_score": similarity,
            "interpretation": interpretation,
            "image1_type": image1_type,
            "image2_type": image2_type,
            "disclaimer": "⚠️ Image comparison is assistive only. Radiological review required.",
        }
        
        logger.info(f"Comparison complete: {interpretation}")
        return result
    
    def cleanup(self):
        """Clean up pipeline and free resources"""
        if self.encoder:
            self.encoder.cleanup()
        
        logger.info("Image Assist Pipeline cleaned up")
