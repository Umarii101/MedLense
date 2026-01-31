"""
GPU memory management utilities for RTX 3080 (10GB VRAM)
Handles model loading, offloading, and memory optimization
"""

import torch
import gc
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Manages GPU memory for efficient model loading on RTX 3080"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.dtype = torch.float16  # Use FP16 for memory efficiency
        
    def _setup_device(self) -> torch.device:
        """Setup CUDA device with validation"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available! Falling back to CPU (VERY SLOW)")
            return torch.device("cpu")
        
        device = torch.device("cuda:0")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        return device
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": total - allocated,
            "utilization_pct": (allocated / total) * 100
        }
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU cache cleared")
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage"""
        stats = self.get_memory_stats()
        if "error" not in stats:
            logger.info(
                f"[{stage}] GPU Memory: "
                f"{stats['allocated_gb']:.2f}GB allocated, "
                f"{stats['free_gb']:.2f}GB free, "
                f"{stats['utilization_pct']:.1f}% utilized"
            )
    
    def prepare_model_kwargs(self, model_size: str = "7b") -> Dict[str, Any]:
        """
        Prepare model loading kwargs optimized for RTX 3080
        
        Args:
            model_size: Model size indicator (7b, 2b, etc.)
        
        Returns:
            Dictionary of model loading arguments
        """
        kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto",  # Auto device mapping
            "low_cpu_mem_usage": True,
        }
        
        # For 7B models on 10GB GPU, we might need 8-bit quantization
        if model_size == "7b" and self.get_available_memory() < 12.0:
            logger.info("Using 8-bit quantization for 7B model on 10GB GPU")
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"
        
        return kwargs
    
    def get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        stats = self.get_memory_stats()
        if "error" in stats:
            return 0.0
        return stats["free_gb"]
    
    def can_load_model(self, estimated_size_gb: float) -> bool:
        """Check if there's enough memory to load a model"""
        available = self.get_available_memory()
        # Keep 1GB buffer for operations
        return available >= (estimated_size_gb + 1.0)
    
    @staticmethod
    def offload_model(model):
        """Offload model from GPU to CPU"""
        if model is not None and hasattr(model, 'cpu'):
            model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def optimize_model_for_inference(model):
        """Optimize model for inference (disable gradients, set eval mode)"""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model


# Global memory manager instance
_memory_manager: Optional[GPUMemoryManager] = None


def get_memory_manager() -> GPUMemoryManager:
    """Get or create global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = GPUMemoryManager()
    return _memory_manager


def check_cuda_setup() -> Dict[str, Any]:
    """Comprehensive CUDA setup check"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
        info["memory_stats"] = get_memory_manager().get_memory_stats()
    
    return info
