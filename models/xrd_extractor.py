import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class XRDFeatureExtractor(nn.Module):
    """
    Converts raw XRD data (Feature Vector) to an input vector for the 
    bridge layer via an MLP.

    This module operates as part of the overall pipeline and updates its 
    parameters via backpropagated gradients from the final loss.
    """
    def __init__(self, input_dim: int = 128, output_dim: int = 128, hidden_dim: Optional[int] = None):
        """
        Args:
            input_dim (int): Dimension of the input XRD vector.
            output_dim (int): Dimension of the output feature vector (to be passed to the bridge).
            hidden_dim (Optional[int]): Dimension of the MLP's hidden layer. 
                                        If None, it defaults to output_dim * 2.
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = output_dim * 2
        
        # Define the feature extractor with a simple MLP structure.
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): An XRD intensity tensor with shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: The final feature vector with shape (batch_size, output_dim).
        """
        # L2 normalization helps stabilize training by scaling the input vectors.
        normalized_x = F.normalize(x, p=2, dim=1)
        
        # Generate the final feature vector by passing through the MLP.
        features = self.extractor(normalized_x)
        
        return features

# --- Code for independent testing ---
# This file can be executed directly to verify that the module works correctly.
# (e.g., `python models/xrd_extractor.py`)
if __name__ == '__main__':
    # Test configurations
    batch_size = 32
    feature_dim = 128
    
    # 1. Create an instance of the model
    print("1. Creating an instance of the XRDFeatureExtractor model.")
    model = XRDFeatureExtractor(input_dim=feature_dim, output_dim=feature_dim)
    print(model)
    
    # 2. Create dummy input data
    print("\n2. Creating dummy data for testing.")
    dummy_input = torch.randn(batch_size, feature_dim)
    
    # 3. Perform a forward pass
    print("\n3. Passing data through the model to check the output.")
    output = model(dummy_input)
    
    # 4. Check the input and output shapes
    print(f"\n--- Test Results ---")
    print(f"Input Data Shape: {dummy_input.shape}")
    print(f"Output Data Shape: {output.shape}")
    
    # 5. Final verification
    if output.shape == (batch_size, feature_dim):
        print("\n✅ Test Successful: Input and output shapes match.")
    else:
        print("\n❌ Test Failed: Input and output shapes do not match.")
