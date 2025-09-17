"""Configuration parser to convert argparse to config objects."""

import argparse
from config import ModelConfig, TrainerConfig, ModalityConfig


def create_configs_from_args(args: argparse.Namespace) -> tuple[ModelConfig, TrainerConfig, ModalityConfig]:
    """Create configuration objects from argparse arguments.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        Tuple of (ModelConfig, TrainerConfig, ModalityConfig)
    """
    # Create model config
    model_config = ModelConfig(
        model_type=getattr(args, 'model_type', 'cgcnn'),
        atom_fea_len=getattr(args, 'atom_fea_len', 64),
        n_conv=getattr(args, 'n_conv', 3),
        h_fea_len=getattr(args, 'h_fea_len', 128),
        n_h=getattr(args, 'n_h', 1),
        graph_type=getattr(args, 'graph_type', 'cgcnn')
    )
    
    # Create trainer config
    trainer_config = TrainerConfig(
        epochs=getattr(args, 'epochs', 30),
        batch_size=getattr(args, 'batch_size', 256),
        learning_rate=getattr(args, 'lr', 0.01),
        optimizer=getattr(args, 'optim', 'SGD'),
        momentum=getattr(args, 'momentum', 0.9),
        weight_decay=getattr(args, 'weight_decay', 0.0),
        lr_milestones=getattr(args, 'lr_milestones', [100]),
        num_workers=getattr(args, 'workers', 0),
        train_ratio=getattr(args, 'train_ratio', None),
        train_size=getattr(args, 'train_size', None),
        val_ratio=getattr(args, 'val_ratio', 0.1),
        val_size=getattr(args, 'val_size', None),
        test_ratio=getattr(args, 'test_ratio', 0.1),
        test_size=getattr(args, 'test_size', None),
        start_epoch=getattr(args, 'start_epoch', 0),
        print_freq=getattr(args, 'print_freq', 10),
        resume=getattr(args, 'resume', ''),
        cuda=getattr(args, 'cuda', True),
        task=getattr(args, 'task', 'regression'),
        best_mae_error=getattr(args, 'best_mae_error', 1e10)
    )
    
    # Create modality config
    modality_config = ModalityConfig(
        use_xrd=getattr(args, 'xrd', False),
        use_text=getattr(args, 'text', False)
    )
    
    return model_config, trainer_config, modality_config