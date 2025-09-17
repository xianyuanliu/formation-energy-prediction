"""Configuration for different modalities (XRD, text, etc)."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModalityConfig:
    """Configuration for multimodal features."""
    # XRD configuration
    use_xrd: bool = False
    xrd_input_dim: int = 128
    xrd_output_dim: int = 64
    xrd_hidden_dim: int = 128
    
    # Text configuration  
    use_text: bool = False
    text_input_dim: int = 384
    text_output_dim: int = 64
    text_hidden_dim: int = 128
    
    def get_total_extra_features(self) -> int:
        """Calculate total extra feature dimensions from modalities."""
        total = 0
        if self.use_xrd:
            total += self.xrd_output_dim
        if self.use_text:
            total += self.text_output_dim
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'xrd': self.use_xrd,
            'text': self.use_text
        }