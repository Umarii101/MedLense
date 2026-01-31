"""
Medical image encoder using RAD-DINO or CLIP.
Provides visual feature extraction for medical images (assistive only).
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict
import logging

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("timm not installed - image encoding may be limited")

from transformers import CLIPProcessor, CLIPModel
from utils.memory import get_memory_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageEncoder:
    """
    Medical image encoder for feature extraction.
    Uses RAD-DINO (preferred) or CLIP as fallback.
    
    ⚠️ ASSISTIVE ONLY - Not for radiological diagnosis
    """
    
    # Model options
    RAD_DINO_MODEL = "facebook/dinov2-base"  # Base DINO model (RAD-DINO uses similar arch)
    CLIP_MODEL = "openai/clip-vit-large-patch14"
    
    def __init__(
        self,
        model_type: str = "clip",  # "dino" or "clip"
        image_size: int = 224
    ):
        """
        Initialize medical image encoder.
        
        Args:
            model_type: "dino" for DINOv2, "clip" for CLIP
            image_size: Input image size
        """
        self.model_type = model_type
        self.image_size = image_size
        self.memory_manager = get_memory_manager()
        self.device = self.memory_manager.device
        
        self.model = None
        self.processor = None
        self.embedding_dim = None
        
        logger.info(f"Initializing Medical Image Encoder ({model_type})...")
        self._load_model()
    
    def _load_model(self):
        """Load image encoding model"""
        try:
            self.memory_manager.log_memory_usage("Before image encoder load")
            
            if self.model_type == "dino" and TIMM_AVAILABLE:
                self._load_dino_model()
            else:
                self._load_clip_model()
            
            self.memory_manager.log_memory_usage("After image encoder load")
            logger.info(f"Image encoder loaded successfully! Embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading image encoder: {e}")
            raise
    
    def _load_dino_model(self):
        """Load DINOv2 model (similar to RAD-DINO architecture)"""
        try:
            logger.info("Loading DINOv2 model...")
            
            # Load DINOv2 via timm
            self.model = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # DINOv2 base has 768-dim embeddings
            self.embedding_dim = 768
            
            # Create simple preprocessing
            from torchvision import transforms
            self.processor = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("DINOv2 model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load DINO model: {e}. Falling back to CLIP...")
            self._load_clip_model()
    
    def _load_clip_model(self):
        """Load CLIP model as fallback"""
        logger.info("Loading CLIP model...")
        
        self.model = CLIPModel.from_pretrained(
            self.CLIP_MODEL,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
        self.processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL)
        
        self.model.eval()
        
        # CLIP ViT-L has 768-dim embeddings
        self.embedding_dim = 768
        
        logger.info("CLIP model loaded successfully")
    
    def encode_image(
        self,
        image: Image.Image,
        return_attention: bool = False
    ) -> Dict:
        """
        Encode medical image to embedding vector.
        
        Args:
            image: PIL Image
            return_attention: Whether to return attention maps (DINO only)
            
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            with torch.no_grad():
                if self.model_type == "dino" and hasattr(self.processor, '__call__'):
                    # DINOv2 processing
                    pixel_values = self.processor(image).unsqueeze(0).to(self.device)
                    outputs = self.model(pixel_values)
                    
                    # Get CLS token embedding
                    if isinstance(outputs, torch.Tensor):
                        embedding = outputs
                    else:
                        embedding = outputs.last_hidden_state[:, 0]  # CLS token
                    
                else:
                    # CLIP processing
                    inputs = self.processor(
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get image features
                    outputs = self.model.get_image_features(**inputs)
                    embedding = outputs
            
            # Normalize embedding
            embedding = F.normalize(embedding, p=2, dim=-1)
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy()
            
            result = {
                "embedding": embedding_np,
                "embedding_shape": embedding_np.shape,
                "embedding_dim": self.embedding_dim,
                "model_type": self.model_type,
                "image_size": image.size,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def encode_image_from_path(self, image_path: str) -> Dict:
        """Encode image from file path"""
        try:
            image = Image.open(image_path)
            return self.encode_image(image)
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            raise
    
    def get_visual_features_for_text(
        self,
        image: Image.Image,
        text_prompts: Optional[list] = None
    ) -> Dict:
        """
        Get visual features with optional text-guided attention (CLIP only).
        
        Args:
            image: PIL Image
            text_prompts: Optional list of text descriptions for guided features
            
        Returns:
            Dictionary with features and similarities
        """
        if self.model_type != "clip":
            logger.warning("Text-guided features only available with CLIP model")
            return self.encode_image(image)
        
        try:
            # Default medical imaging prompts if none provided
            if text_prompts is None:
                text_prompts = [
                    "normal medical scan",
                    "abnormal findings",
                    "clear image quality",
                    "poor image quality"
                ]
            
            with torch.no_grad():
                # Process image and text
                inputs = self.processor(
                    text=text_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Get features
                outputs = self.model(**inputs)
                
                # Calculate similarities
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get image embedding
            image_features = outputs.image_embeds
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            result = {
                "embedding": image_features.cpu().numpy(),
                "embedding_shape": image_features.shape,
                "text_similarities": {
                    prompt: float(prob)
                    for prompt, prob in zip(text_prompts, probs[0])
                },
                "model_type": self.model_type,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in text-guided encoding: {e}")
            # Fallback to standard encoding
            return self.encode_image(image)
    
    def assess_image_quality(self, image: Image.Image) -> Dict:
        """
        Assess medical image quality using heuristics.
        
        Returns:
            Dictionary with quality metrics
        """
        # Convert to numpy
        img_array = np.array(image)
        
        # Basic quality metrics
        metrics = {
            "size": image.size,
            "mode": image.mode,
            "mean_intensity": float(np.mean(img_array)),
            "std_intensity": float(np.std(img_array)),
            "dynamic_range": float(np.max(img_array) - np.min(img_array)),
        }
        
        # Quality assessment
        quality_issues = []
        
        # Check resolution
        min_size = min(image.size)
        if min_size < 224:
            quality_issues.append(f"Low resolution: {image.size}")
        
        # Check contrast
        if metrics["std_intensity"] < 20:
            quality_issues.append("Low contrast detected")
        
        # Check if too dark or too bright
        if metrics["mean_intensity"] < 30:
            quality_issues.append("Image may be too dark")
        elif metrics["mean_intensity"] > 225:
            quality_issues.append("Image may be too bright")
        
        metrics["quality_issues"] = quality_issues
        metrics["quality_assessment"] = "Good" if not quality_issues else "Issues detected"
        
        return metrics
    
    def cleanup(self):
        """Clean up model and free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.memory_manager.clear_cache()
        logger.info("Image encoder cleaned up")


def create_medical_image_description(
    image_features: Dict,
    quality_metrics: Dict
) -> str:
    """
    Create text description of image features for LLM context.
    
    Args:
        image_features: Output from encode_image
        quality_metrics: Output from assess_image_quality
        
    Returns:
        Text description for LLM
    """
    description = f"""Medical Image Analysis:
- Image size: {quality_metrics['size']}
- Quality: {quality_metrics['quality_assessment']}
- Embedding dimension: {image_features['embedding_dim']}
- Model used: {image_features['model_type']}
"""
    
    if quality_metrics['quality_issues']:
        description += f"\nQuality concerns:\n"
        for issue in quality_metrics['quality_issues']:
            description += f"  - {issue}\n"
    
    if 'text_similarities' in image_features:
        description += "\nVisual characteristics:\n"
        for prompt, sim in image_features['text_similarities'].items():
            if sim > 0.3:  # Only include high similarity
                description += f"  - {prompt}: {sim:.2%}\n"
    
    return description
