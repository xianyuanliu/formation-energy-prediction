"""Model factory for creating different model types."""

import torch
import torch.nn as nn
from typing import Dict, Type, Callable, Any

from models.cgcnn import CrystalGraphConvNet
from config import ModelConfig, ModalityConfig


class ModelRegistry:
    """Registry for different model types."""
    
    def __init__(self):
        self._models: Dict[str, Callable] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model types."""
        self.register("cgcnn", CrystalGraphConvNet)
        # Future models can be registered here
        # self.register("transformer", TransformerModel)
        # self.register("attention_gnn", AttentionGNN)
    
    def register(self, name: str, model_class: Callable):
        """Register a new model type."""
        self._models[name] = model_class
    
    def get_model_class(self, name: str) -> Callable:
        """Get model class by name."""
        if name not in self._models:
            raise ValueError(f"Unknown model type: {name}. Available: {list(self._models.keys())}")
        return self._models[name]
    
    def list_models(self) -> list:
        """List all available model types."""
        return list(self._models.keys())


class ModelFactory:
    """Factory for creating models with different configurations."""
    
    def __init__(self):
        self.registry = ModelRegistry()
    
    def create_model(
        self, 
        model_config: ModelConfig, 
        modality_config: ModalityConfig,
        orig_atom_fea_len: int,
        nbr_fea_len: int
    ) -> nn.Module:
        """Create a model instance based on configuration.
        
        Args:
            model_config: Model architecture configuration
            modality_config: Modality configuration (XRD, text, etc)
            orig_atom_fea_len: Original atom feature length from data
            nbr_fea_len: Neighbor feature length from data
            
        Returns:
            Configured model instance
        """
        model_class = self.registry.get_model_class(model_config.model_type)
        
        # Update model config with data-dependent dimensions
        model_config.orig_atom_fea_len = orig_atom_fea_len
        model_config.nbr_fea_len = nbr_fea_len
        
        # Create model with all configuration
        model_params = {
            'orig_atom_fea_len': orig_atom_fea_len,
            'nbr_fea_len': nbr_fea_len,
            **model_config.to_dict(),
            **modality_config.to_dict()
        }
        
        return model_class(**model_params)
    
    def register_model(self, name: str, model_class: Callable):
        """Register a new model type."""
        self.registry.register(name, model_class)
    
    def list_available_models(self) -> list:
        """List all available model types."""
        return self.registry.list_models()


# Global model factory instance
model_factory = ModelFactory()