import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np
import pandas as pd  # Missing import!
import os
from pathlib import Path

class TextEmbeddingDataset(Dataset):
    """
    A custom PyTorch Dataset to load space group text embeddings from a .csv file.

    This dataset is designed for an unsupervised or multi-modal setup.
    It loads pre-computed 384-dimensional text embeddings from Sentence-BERT
    and follows the exact same logic as XRDDataset for consistency.
    """
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path (str): The path to the SG_text_data.csv file.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The file '{csv_path}' was not found.")
        
        # Load CSV file with pandas (same as XRDDataset)
        df = pd.read_csv(csv_path)

        # Check for required 'Composition' column (same as XRDDataset)
        if 'Composition' not in df.columns:
            raise KeyError("The required column 'Composition' was not found in the CSV file.")

        try:
            # Find the starting column index for embeddings (equivalent to 'xrd_0' in XRDDataset)
            emb_start_col_index = df.columns.get_loc('emb_000')
        except KeyError:
            raise KeyError("The required column 'emb_000' was not found in the CSV file.")

        # Find consecutive embedding columns (same logic as XRDDataset)
        remaining_columns = df.columns[emb_start_col_index:]
        is_emb_col = remaining_columns.str.startswith('emb_')
        consecutive_mask = is_emb_col.cumprod().astype(bool)
        
        # Count the number of consecutive embedding features
        self.num_text_features = consecutive_mask.sum()

        if self.num_text_features == 0:
            raise ValueError("Found 'emb_000' but no subsequent 'emb_n' columns to form a feature set.")

        # Extract embedding values and compositions (same logic as XRDDataset)
        emb_end_col_index = emb_start_col_index + self.num_text_features
        emb_values = df.iloc[:, emb_start_col_index:emb_end_col_index].values
        compositions = df['Composition'].astype(str).values 
        
        # Store embeddings as composition-keyed dictionary (identical to XRDDataset structure)
        self.text_embeddings = {
            comp: torch.tensor(feat, dtype=torch.float32)
            for comp, feat in zip(compositions, emb_values)
        }
        
        # Also store space group information for additional functionality
        if 'space_group' in df.columns:
            space_groups = df['space_group'].astype(str).values
            self.space_groups = {
                comp: sg
                for comp, sg in zip(compositions, space_groups)
            }
        else:
            self.space_groups = {}

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.text_embeddings)

    def __getitem__(self, key) -> torch.Tensor:
        """
        Retrieves a single sample of text embeddings from the dataset.
        
        Args:
            key: Can be either int (index) for DataLoader compatibility 
                 or str (composition) for direct lookup
            
        Returns:
            torch.Tensor: The text embedding tensor for the composition
        """
        if isinstance(key, int):
            # For DataLoader compatibility - convert index to composition
            compositions = list(self.text_embeddings.keys())
            if key >= len(compositions):
                raise IndexError(f"Index {key} out of range for dataset of size {len(compositions)}")
            composition = compositions[key]
            return self.text_embeddings[composition]
        elif isinstance(key, str):
            # Direct composition lookup
            if key not in self.text_embeddings:
                raise KeyError(f"Composition '{key}' not found in dataset")
            return self.text_embeddings[key]
        else:
            raise TypeError(f"Key must be int or str, got {type(key)}")
    
    def get_space_group(self, composition: str) -> str:
        """
        Retrieves the space group for a given composition.
        
        Args:
            composition (str): The composition string to lookup
            
        Returns:
            str: The space group for the composition
        """
        if composition in self.space_groups:
            return self.space_groups[composition]
        else:
            raise KeyError(f"Space group information not available for composition: {composition}")
    
    def get_compositions(self) -> list:
        """
        Returns a list of all available compositions in the dataset.
        
        Returns:
            list: List of composition strings
        """
        return list(self.text_embeddings.keys())
    
    def has_composition(self, composition: str) -> bool:
        """
        Checks if a composition exists in the dataset.
        
        Args:
            composition (str): The composition string to check
            
        Returns:
            bool: True if composition exists, False otherwise
        """
        return composition in self.text_embeddings

class TextFeatureExtractor(nn.Module):
    """
    Converts pre-computed text embeddings (384-dimensional Sentence-BERT vectors) 
    to an input vector for the bridge layer via an MLP.
    """
    def __init__(self, input_dim: int = 384, output_dim: int = 128, hidden_dim: Optional[int] = None):
        """
        Args:
            input_dim (int): Dimension of the input text embedding vector. 
                           Defaults to 384 for Sentence-BERT all-MiniLM-L6-v2.
            output_dim (int): Dimension of the output feature vector (to be passed to the bridge).
            hidden_dim (Optional[int]): Dimension of the MLP's hidden layer. 
                                        If None, it defaults to output_dim * 2.
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = output_dim * 2
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A text embedding tensor with shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: The final feature vector with shape (batch_size, output_dim).
        """
        # Validate input shape
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(1)}")
        
        # Text embeddings are already normalized from Sentence-BERT, but we can re-normalize
        normalized_x = F.normalize(x, p=2, dim=1)
        features = self.extractor(normalized_x)
        return features

# --- Code for independent testing ---
if __name__ == '__main__':
    # Prepare output file
    output_file = "sg_text_module_output.txt"
    output_lines = []
    
    def log_print(message):
        """Print message and save to output list"""
        print(message)
        output_lines.append(message)
    
    try:
        # 1. Load data using the Dataset.
        log_print("1. Loading data using TextEmbeddingDataset...")
        
        # Try multiple possible paths for the CSV file
        possible_paths = [
            'data/SG_text_data.csv',
            'SG_text_data.csv',
            '../data/SG_text_data.csv',
            './SG_text_data.csv'
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError(f"SG_text_data.csv not found in any of these paths: {possible_paths}")
        
        log_print(f"   Found embeddings file at: {csv_path}")
        dataset = TextEmbeddingDataset(csv_path=csv_path)
        
        # Get the dynamically detected feature dimension from the dataset.
        detected_feature_dim = dataset.num_text_features
        log_print(f"   ...Successfully loaded {len(dataset)} samples.")
        log_print(f"   ...Detected {detected_feature_dim} features per sample.")

        # 2. Create an instance of the model with the correct input dimension.
        log_print("\n2. Creating an instance of the TextFeatureExtractor model.")
        model = TextFeatureExtractor(input_dim=detected_feature_dim, output_dim=128)
        
        # Convert model architecture to string for logging
        model_str = str(model)
        log_print(f"Model architecture:\n{model_str}")
        
        # 3. Use DataLoader to create a batch of real data for testing.
        log_print("\n3. Creating a DataLoader to batch the real data.")
        data_loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=False, num_workers=0)
        real_data_batch = next(iter(data_loader)) # Get one batch
        
        # 4. Perform a forward pass with the real data batch.
        log_print("\n4. Passing a batch of real data through the model.")
        with torch.no_grad():  # Add no_grad for testing
            output = model(real_data_batch)
        
        # 5. Check the input and output shapes.
        log_print(f"\n--- Test Results ---")
        log_print(f"Input Data Shape: {real_data_batch.shape}")
        log_print(f"Output Data Shape: {output.shape}")
        log_print(f"Input Data Range: [{real_data_batch.min():.4f}, {real_data_batch.max():.4f}]")
        log_print(f"Output Data Range: [{output.min():.4f}, {output.max():.4f}]")
        
        # 6. Show some statistics about the embeddings
        log_print(f"\n--- Embedding Statistics ---")
        log_print(f"Input Mean: {real_data_batch.mean():.4f}")
        log_print(f"Input Std: {real_data_batch.std():.4f}")
        log_print(f"Output Mean: {output.mean():.4f}")
        log_print(f"Output Std: {output.std():.4f}")
        
        # 7. Final verification.
        expected_output_shape = (real_data_batch.shape[0], 128)
        if output.shape == expected_output_shape:
            log_print("\n✅ Test Successful: Model processed the text embeddings and produced the correct output shape.")
        else:
            log_print(f"\n❌ Test Failed: Expected output shape {expected_output_shape}, got {output.shape}")
            
        # 8. Test with different batch sizes
        log_print("\n5. Testing with different batch sizes...")
        for batch_size in [1, 16, 64]:
            if batch_size <= len(dataset):
                test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                test_batch = next(iter(test_loader))
                with torch.no_grad():
                    test_output = model(test_batch)
                log_print(f"   Batch size {batch_size}: {test_batch.shape} → {test_output.shape} ✅")
        
        # 9. Test direct composition access
        log_print("\n6. Testing direct composition access...")
        compositions = dataset.get_compositions()
        if len(compositions) > 0:
            test_comp = compositions[0]
            direct_embedding = dataset[test_comp]
            log_print(f"   Direct access to '{test_comp}': {direct_embedding.shape} ✅")
            
            # Test space group access
            try:
                space_group = dataset.get_space_group(test_comp)
                log_print(f"   Space group for '{test_comp}': {space_group} ✅")
            except KeyError:
                log_print(f"   Space group info not available for '{test_comp}' ⚠️")

    except (FileNotFoundError, ValueError, KeyError) as e:
        log_print(f"\n❌ An error occurred during testing: {e}")
        log_print("\n💡 Make sure:")
        log_print("   1. SG_text_data.csv file is available in the working directory")
        log_print("   2. The CSV file contains 'Composition', 'space_group', and 'emb_000' columns")
        log_print("   3. Embedding columns are properly formatted (emb_000, emb_001, etc.)")
    
    except Exception as e:
        log_print(f"\n❌ Unexpected error occurred: {e}")
        import traceback
        log_print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Save output to txt file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Text Module Test Output\n")
                f.write("=" * 50 + "\n\n")
                for line in output_lines:
                    f.write(line + "\n")
            print(f"\n📄 Test output saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save output file: {e}")