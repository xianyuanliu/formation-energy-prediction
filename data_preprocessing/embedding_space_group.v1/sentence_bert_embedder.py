import json
import numpy as np
import pandas as pd
import os

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
        """Load space group fingerprints from JSON file"""
        try:
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"File '{json_file}' not found in current directory")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                fingerprints = json.load(f)
            
            print(f"📁 Loaded {len(fingerprints)} space group descriptions")
            return fingerprints
            
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return None
    
    def generate_embeddings(self, fingerprints):
        """Generate 384-dimensional embeddings using Sentence-BERT"""
        if self.model is None:
            print("❌ Model not loaded")
            return None, None
        
        # Extract symbols and texts
        symbols = list(fingerprints.keys())
        texts = list(fingerprints.values())
        
        print(f"🔢 Generating embeddings for {len(texts)} texts...")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            
            print(f"✓ Generated embeddings with shape: {embeddings.shape}")
            
            # Store results
            self.embeddings = embeddings
            self.space_group_symbols = symbols
            
            return embeddings, symbols
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return None, None
    
    def save_csv(self, output_file: str = "spacegroup_embeddings_384d.csv"):
        """Save embeddings as CSV with space_group as first column"""
        if self.embeddings is None or self.space_group_symbols is None:
            print("❌ No embeddings to save")
            return False
        
        try:
            print(f"💾 Creating CSV file: {output_file}")
            
            # Create DataFrame with space_group as first column
            data = {'space_group': self.space_group_symbols}
            
            # Add embedding columns
            for i in range(self.embeddings.shape[1]):
                col_name = f'emb_{i:03d}'
                data[col_name] = self.embeddings[:, i]
            
            df = pd.DataFrame(data)
            
            # Save CSV
            df.to_csv(output_file, index=False)
            
            # Verify and show info
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024 / 1024
                print(f"✓ CSV created: {output_file}")
                print(f"   Shape: {df.shape}")
                print(f"   Size: {file_size:.2f} MB")
                print(f"   Columns: space_group + emb_000 to emb_383")
                return True
            else:
                print(f"❌ Failed to create: {output_file}")
                return False
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return False
    
    def run_pipeline(self):
        """Run the complete pipeline to generate CSV"""
        print("🚀 Space Group Embeddings → CSV Pipeline")
        print("=" * 50)
        
        # Load model
        if not self.load_sentence_transformer():
            return False
        
        # Load fingerprints
        fingerprints = self.load_fingerprints()
        if fingerprints is None:
            return False
        
        # Generate embeddings
        embeddings, symbols = self.generate_embeddings(fingerprints)
        if embeddings is None:
            return False
        
        # Save CSV
        if not self.save_csv():
            return False
        
        print("\n✅ Pipeline completed!")
        print("📄 spacegroup_embeddings_384d.csv ready for use")
        
        return True

def main():
    """Main execution"""
    embedder = SentenceBERTEmbedder()
    
    # Check if JSON exists
    if not os.path.exists("spacegroup_fingerprints.json"):
        print("❌ spacegroup_fingerprints.json not found")
        return
    
    # Run pipeline
    embedder.run_pipeline()

if __name__ == "__main__":
    main()