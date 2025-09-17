"""Configuration classes for the formation energy prediction system."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = "cgcnn"
    atom_fea_len: int = 64
    n_conv: int = 3
    h_fea_len: int = 128
    n_h: int = 1
    graph_type: str = "cgcnn"
    
    # Feature dimensions (these will be set at runtime based on data)
    orig_atom_fea_len: Optional[int] = None
    nbr_fea_len: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'atom_fea_len': self.atom_fea_len,
            'n_conv': self.n_conv,
            'h_fea_len': self.h_fea_len,
            'n_h': self.n_h,
            'graph_type': self.graph_type
        }