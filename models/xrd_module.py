import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import pandas as pd

class XRDDataset(Dataset):
    """
    A custom PyTorch Dataset to load XRD feature data from a CSV file.

    This dataset is designed for an unsupervised or multi-modal setup.
    It robustly finds the 'xrd_0' column and then automatically determines
    the number of consecutive 'xrd_n' columns to use as the feature set.
    """
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path (str): The path to the XRD_data.csv file.
        """
        df = pd.read_csv(csv_path)

        try:
            xrd_start_col_index = df.columns.get_loc('xrd_0')
        except KeyError:
            raise KeyError("The required column 'xrd_0' was not found in the CSV file.")

        remaining_columns = df.columns[xrd_start_col_index:]
        is_xrd_col = remaining_columns.str.startswith('xrd_')
        consecutive_mask = is_xrd_col.cumprod().astype(bool)
        
        self.num_xrd_features = consecutive_mask.sum()

        if self.num_xrd_features == 0:
            raise ValueError("Found 'xrd_0' but no subsequent 'xrd_n' columns to form a feature set.")

        xrd_end_col_index = xrd_start_col_index + self.num_xrd_features
        xrd_features = df.iloc[:, xrd_start_col_index:xrd_end_col_index].values
        self.xrd_features = torch.tensor(xrd_features, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.xrd_features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves a single sample of XRD features from the dataset at the given index."""
        return self.xrd_features[idx]

class XRDFeatureExtractor(nn.Module):
    """
    Converts raw XRD data (Feature Vector) to an input vector for the 
    bridge layer via an MLP.
    """
    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: Optional[int] = None):
        """
        Args:
            input_dim (int): Dimension of the input XRD vector. Should match the number of features from the dataset.
            output_dim (int): Dimension of the output feature vector (to be passed to the bridge).
            hidden_dim (Optional[int]): Dimension of the MLP's hidden layer. 
                                        If None, it defaults to output_dim * 2.
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = output_dim * 2
        
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
        normalized_x = F.normalize(x, p=2, dim=1)
        features = self.extractor(normalized_x)
        return features

# --- Code for independent testing ---
if __name__ == '__main__':
    try:
        # 1. Load data using the Dataset.
        print("1. Loading data using XRDDataset...")
        # Make sure the path is correct for your project structure.
        dataset = XRDDataset(csv_path='data/XRD_data.csv')
        
        # Get the dynamically detected feature dimension from the dataset.
        detected_feature_dim = dataset.num_xrd_features
        print(f"   ...Successfully loaded {len(dataset)} samples.")
        print(f"   ...Detected {detected_feature_dim} features per sample.")

        # 2. Create an instance of the model with the correct input dimension.
        print("\n2. Creating an instance of the XRDFeatureExtractor model.")
        model = XRDFeatureExtractor(input_dim=detected_feature_dim, output_dim=128)
        print(model)
        
        # 3. Use DataLoader to create a batch of real data for testing.
        print("\n3. Creating a DataLoader to batch the real data.")
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        real_data_batch = next(iter(data_loader)) # Get one batch
        
        # 4. Perform a forward pass with the real data batch.
        print("\n4. Passing a batch of real data through the model.")
        output = model(real_data_batch)
        
        # 5. Check the input and output shapes.
        print(f"\n--- Test Results ---")
        print(f"Input Data Shape: {real_data_batch.shape}")
        print(f"Output Data Shape: {output.shape}")
        
        # 6. Final verification.
        if output.shape == (real_data_batch.shape[0], 128):
            print("\n✅ Test Successful: Model processed the data and produced the correct output shape.")
        else:
            print("\n❌ Test Failed: Output shape is incorrect.")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\n❌ An error occurred during testing: {e}")