"""Training configuration classes."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TrainerConfig:
    """Configuration for training parameters."""
    # Basic training parameters
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.01
    optimizer: str = "SGD"
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_milestones: List[int] = None
    
    # Data parameters
    num_workers: int = 0
    train_ratio: Optional[float] = None
    train_size: Optional[int] = None
    val_ratio: float = 0.1
    val_size: Optional[int] = None
    test_ratio: float = 0.1
    test_size: Optional[int] = None
    
    # Training control
    start_epoch: int = 0
    print_freq: int = 10
    resume: str = ""
    cuda: bool = True
    
    # Task specific
    task: str = "regression"
    best_mae_error: float = 1e10
    
    def __post_init__(self):
        if self.lr_milestones is None:
            self.lr_milestones = [100]