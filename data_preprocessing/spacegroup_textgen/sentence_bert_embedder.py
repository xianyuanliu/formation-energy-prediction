import json
import numpy as np
import os
from pathlib import Path
import pickle
from datetime import datetime

class SentenceBERTEmbedder:
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.space_group_symbols = None
        self.model_name = 'all-MiniLM-L6-v2'  # 384 dimensions
        
    def load_sentence_transformer(self):
        """Load Sentence-BERT model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🤖 Loading Sentence-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✓ Model loaded successfully (embedding dimension: {self.model.get_sentence_embedding_dimension()})")
            return True
        except ImportError:
            print("❌ Error: sentence-transformers not installed")
            print("   Please install: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_fingerprints(self, json_file: str = "spacegroup_fingerprints.json"):
        """Load space group fingerprints from JSON file in current directory"""
        try:
            # Check if file exists in current directory
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"File '{json_file}' not found in current directory")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                fingerprints = json.load(f)
            
            print(f"📁 Loaded {len(fingerprints)} space group descriptions from '{json_file}'")
            
            # Verify we have 230 entries
            if len(fingerprints) != 230:
                print(f"⚠️  Warning: Expected 230 space groups, found {len(fingerprints)}")
            
            return fingerprints
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("   Make sure 'spacegroup_fingerprints.json' exists in the current directory")
            return None
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return None
    
    def generate_embeddings(self, fingerprints):
        """Generate 384-dimensional embeddings using Sentence-BERT"""
        if self.model is None:
            print("❌ Model not loaded. Call load_sentence_transformer() first.")
            return None, None
        
        # Extract symbols and texts
        symbols = list(fingerprints.keys())
        texts = list(fingerprints.values())
        
        print(f"🔢 Generating 384-dimensional embeddings for {len(texts)} texts...")
        print("   This may take a few moments...")
        
        try:
            # Generate embeddings with progress bar
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=True, 
                convert_to_numpy=True,
                batch_size=32  # Process in batches for efficiency
            )
            
            print(f"✓ Successfully generated embeddings with shape: {embeddings.shape}")
            
            # Store results
            self.embeddings = embeddings
            self.space_group_symbols = symbols
            
            return embeddings, symbols
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return None, None
    
    def create_data_directory(self):
        """Create ../../data/ directory if it doesn't exist"""
        try:
            # Get path to ../../data/
            current_dir = Path.cwd()
            data_dir = current_dir.parent.parent / "data"
            
            # Create directory if it doesn't exist
            data_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📁 Data directory: {data_dir}")
            return data_dir
            
        except Exception as e:
            print(f"❌ Error creating data directory: {e}")
            return None
    
    def save_embeddings(self, data_dir):
        """Save embeddings for MLP training"""
        if self.embeddings is None or self.space_group_symbols is None:
            print("❌ No embeddings to save. Generate embeddings first.")
            return False
        
        try:
            # Save only the embeddings array for MLP training
            embeddings_file = data_dir / "spacegroup_embeddings_384d.npy"
            np.save(embeddings_file, self.embeddings)
            print(f"💾 Saved embeddings for MLP: {embeddings_file}")
            
            # Save space group mapping for reference (optional)
            mapping_file = "spacegroup_mapping.txt"
            with open(mapping_file, 'w', encoding='utf-8') as f:
                f.write("Space Group Index Mapping\n")
                f.write("=" * 30 + "\n")
                f.write("Index | Space Group Symbol\n")
                f.write("-" * 30 + "\n")
                for i, symbol in enumerate(self.space_group_symbols):
                    f.write(f"{i:5d} | {symbol}\n")
            
            print(f"📄 Saved index mapping: {mapping_file}")
            
            # Display summary
            print(f"\n📊 Files for MLP Training:")
            print(f"   Main file: {embeddings_file.name}")
            print(f"   Shape: {self.embeddings.shape}")
            print(f"   Data type: {self.embeddings.dtype}")
            print(f"   Size: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete embedding pipeline"""
        print("🚀 Starting Space Group Sentence-BERT Embedding Pipeline")
        print("=" * 60)
        
        # Step 1: Load Sentence-BERT model
        if not self.load_sentence_transformer():
            return False
        
        # Step 2: Load fingerprints from JSON
        fingerprints = self.load_fingerprints()
        if fingerprints is None:
            return False
        
        # Step 3: Generate embeddings
        embeddings, symbols = self.generate_embeddings(fingerprints)
        if embeddings is None:
            return False
        
        # Step 4: Create data directory
        data_dir = self.create_data_directory()
        if data_dir is None:
            return False
        
        # Step 5: Save embeddings
        if not self.save_embeddings(data_dir):
            return False
        
        print("\n✅ Pipeline completed successfully!")
        print(f"   📁 Files saved to: {data_dir}")
        print(f"   🔢 Ready for MLP training: {embeddings.shape[0]} samples × {embeddings.shape[1]} features")
        
        return True

def main():
    """Main execution function"""
    embedder = SentenceBERTEmbedder()
    
    print("Space Group Sentence-BERT Embedder")
    print("Converts 230 space group descriptions to 384-dimensional vectors")
    print()
    
    # Check if JSON file exists before starting
    if not os.path.exists("spacegroup_fingerprints.json"):
        print("❌ Error: 'spacegroup_fingerprints.json' not found in current directory")
        print("   Please make sure the JSON file is in the same directory as this script")
        return
    
    # Run the complete pipeline
    success = embedder.run_complete_pipeline()
    
    if success:
        print("\n🎉 All done! Your space group embeddings are ready for ML training.")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")

if __name__ == "__main__":
    main()