#!/usr/bin/env python3
"""
Example script demonstrating how to extend the modular system with new model types.

This script shows how easy it is now to add new models and modalities 
to the formation energy prediction framework.
"""

import torch
import torch.nn as nn
from config import ModelConfig, TrainerConfig, ModalityConfig
from factories import model_factory


# Example 1: Creating a custom model type
class SimpleLinearModel(nn.Module):
    """A simple linear model for demonstration purposes."""
    
    def __init__(self, orig_atom_fea_len, nbr_fea_len, **kwargs):
        super().__init__()
        # Ignore graph-specific parameters for this simple model
        self.linear = nn.Linear(orig_atom_fea_len, 1)
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, 
                xrd_feature=None, text_feature=None):
        # Simple average pooling over atoms
        batch_size = len(crystal_atom_idx)
        crystal_features = []
        
        for i in range(batch_size):
            atom_indices = crystal_atom_idx[i]
            crystal_atom_fea = atom_fea[atom_indices]
            # Simple average pooling
            pooled = torch.mean(crystal_atom_fea, dim=0)
            crystal_features.append(pooled)
        
        crystal_features = torch.stack(crystal_features)
        output = self.linear(crystal_features)
        return output


def demonstrate_extensibility():
    """Demonstrate how to extend the system with new model types."""
    
    print("🔧 Demonstrating System Extensibility")
    print("=" * 50)
    
    # Step 1: Register a new model type
    print("1. Registering new model type...")
    model_factory.register_model('simple_linear', SimpleLinearModel)
    print(f"   ✓ Available models: {model_factory.list_available_models()}")
    print()
    
    # Step 2: Create configuration for the new model
    print("2. Creating configuration for new model...")
    model_config = ModelConfig(
        model_type='simple_linear',  # Use our new model
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1
    )
    
    modality_config = ModalityConfig(
        use_xrd=False,
        use_text=False
    )
    print(f"   ✓ Model config: {model_config.model_type}")
    print()
    
    # Step 3: Create the model using the factory
    print("3. Creating model instance...")
    try:
        model = model_factory.create_model(
            model_config=model_config,
            modality_config=modality_config,
            orig_atom_fea_len=92,  # Example from real data
            nbr_fea_len=41         # Example from real data
        )
        print(f"   ✓ Successfully created {type(model).__name__}")
        print(f"   ✓ Model has {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    print()
    
    # Step 4: Show how easy it is to switch between model types
    print("4. Demonstrating model type switching...")
    
    for model_type in ['cgcnn', 'simple_linear']:
        config = ModelConfig(model_type=model_type)
        try:
            model = model_factory.create_model(
                model_config=config,
                modality_config=modality_config,
                orig_atom_fea_len=92,
                nbr_fea_len=41
            )
            params = sum(p.numel() for p in model.parameters())
            print(f"   ✓ {model_type}: {type(model).__name__} ({params:,} parameters)")
        except Exception as e:
            print(f"   ✗ {model_type}: Failed - {e}")
    
    print()
    return True


def demonstrate_modality_flexibility():
    """Show how easy it is to configure different modalities."""
    
    print("🎯 Demonstrating Modality Flexibility")
    print("=" * 50)
    
    modality_configs = [
        ("Crystal only", ModalityConfig(use_xrd=False, use_text=False)),
        ("Crystal + XRD", ModalityConfig(use_xrd=True, use_text=False)),
        ("Crystal + Text", ModalityConfig(use_xrd=False, use_text=True)),
        ("All modalities", ModalityConfig(use_xrd=True, use_text=True)),
    ]
    
    model_config = ModelConfig(model_type='cgcnn')
    
    for name, modality_config in modality_configs:
        try:
            model = model_factory.create_model(
                model_config=model_config,
                modality_config=modality_config,
                orig_atom_fea_len=92,
                nbr_fea_len=41
            )
            params = sum(p.numel() for p in model.parameters())
            extra_features = modality_config.get_total_extra_features()
            print(f"   ✓ {name}: {params:,} parameters (+{extra_features} extra features)")
        except Exception as e:
            print(f"   ✗ {name}: Failed - {e}")
    
    print()


def main():
    """Main demonstration function."""
    
    print("🚀 Formation Energy Prediction - Modular System Demo")
    print("=" * 60)
    print()
    print("This script demonstrates the new modular architecture that makes")
    print("it easy to add new model types and configure different modalities.")
    print()
    
    success = True
    
    try:
        if not demonstrate_extensibility():
            success = False
        
        demonstrate_modality_flexibility()
        
        if success:
            print("🎉 All demonstrations completed successfully!")
            print()
            print("Key Benefits of the Modular System:")
            print("• Easy to add new model architectures")
            print("• Flexible modality configuration")
            print("• Type-safe configuration objects")
            print("• Clean separation of concerns")
            print("• Backward compatibility maintained")
            print("• Extensible through registry pattern")
        else:
            print("❌ Some demonstrations failed.")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()