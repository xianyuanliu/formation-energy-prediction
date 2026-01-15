import torch
import pandas as pd
import re
import os
import json
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel

# ==============================================================================
# 1. Verified Space Group Description Generator
#    - Uses strict regex logic to parse Hermann-Mauguin symbols correctly.
#    - Generates natural language descriptions for MatBERT.
# ==============================================================================
class SpaceGroupDescriber:
    def __init__(self):
        # Geometric descriptions for each crystal system
        # Explicitly mentions angle constraints for better model understanding
        self.geometry_desc = {
            'Cubic': "The unit cell forms a perfect cube (a=b=c; all angles are 90°).",
            'Tetragonal': "The unit cell forms a square prism (a=b≠c; all angles are 90°).",
            'Orthorhombic': "The unit cell forms a rectangular prism (a≠b≠c; all angles are 90°).",
            'Hexagonal': "The unit cell forms a hexagonal prism (a=b≠c; two angles are 90° and one is 120°).",
            'Trigonal': "The unit cell possesses 3-fold symmetry (typically hexagonal axes with two 90° and one 120° angle).",
            'Monoclinic': "The unit cell forms a tilted prism (a≠b≠c; two angles are 90° and the third angle is non-90°).",
            'Triclinic': "The unit cell is a general parallelepiped (a≠b≠c; all three angles are generally different and non-90°)."
        }
        
        # Descriptions for Bravais lattices
        self.lattice_desc = {
            'P': "a Primitive lattice (corners only)",
            'I': "a Body-Centered lattice",
            'F': "a Face-Centered lattice",
            'C': "a Base-Centered lattice (C-face)",
            'A': "a Base-Centered lattice (A-face)",
            'R': "a Rhombohedral lattice"
        }

        # Physical definitions of axes for each system
        # Maps pos1, pos2, pos3 to actual crystallographic directions
        self.axis_definitions = {
            'Cubic': ['principal axes <100>', 'body diagonals <111>', 'face diagonals <110>'],
            'Hexagonal': ['c-axis (principal)', 'a-axes (secondary)', 'inter-axial directions'],
            'Trigonal': ['c-axis (principal)', 'a-axes (secondary)', 'inter-axial directions'],
            'Tetragonal': ['c-axis (principal)', 'a/b axes', 'face diagonals'],
            'Orthorhombic': ['x-axis (a)', 'y-axis (b)', 'z-axis (c)'],
            'Monoclinic': ['unique axis (b)', 'perpendicular plane', ''],
            'Triclinic': ['general direction', '', '']
        }

    def generate_description(self, symbol):
        """
        Generates a full natural language description from a space group symbol.
        """
        if pd.isna(symbol) or not isinstance(symbol, str): return ""
        
        # 1. Parsing Strategy
        # Use regex [1-6] to strictly match single-digit rotation axes.
        # This prevents bugs where '43' is read as a single number in 'F-43m'.
        bravais = symbol[0]
        tokens = re.findall(r"(-?[1-6](?:_[1-6])?(?:/[a-z])?)|([a-z])", symbol[1:].strip())
        # Flatten tuple results from findall
        tokens = [m[0] if m[0] else m[1] for m in tokens]

        # 2. Identify Crystal System based on token positions
        system = self._identify_system(tokens)
        
        # 3. Assemble the Text Description
        desc = f"Space group {symbol} belongs to the {system} system. "
        desc += f"{self.geometry_desc.get(system, '')} "
        desc += f"It features {self.lattice_desc.get(bravais, 'an unknown lattice')}. "
        
        # Map tokens to specific axes (e.g., pos1 -> c-axis for Hexagonal)
        axis_names = self.axis_definitions.get(system, ['axis 1', 'axis 2', 'axis 3'])
        
        elements = []
        for i, token in enumerate(tokens[:3]): # Handle up to 3 positions
            direction = axis_names[i] if i < len(axis_names) else "general direction"
            explanation = self._explain_token(token)
            if explanation:
                elements.append(f"along {direction}, there is {explanation}")
        
        if elements:
            desc += "Symmetry elements: " + "; ".join(elements) + "."
        
        return desc

    def _identify_system(self, tokens):
        """
        Determines the crystal system using standard crystallographic rules.
        """
        p1 = tokens[0] if len(tokens) > 0 else ""
        p2 = tokens[1] if len(tokens) > 1 else ""
        
        # Cubic: 3-fold axis in the second position (e.g., Fm-3m)
        if p2 and ('3' in p2.split('/')[0]): return "Cubic"
        
        # Check the principal axis (first position)
        base_p1 = p1.replace('-', '')
        if base_p1.startswith('6'): return "Hexagonal"
        if base_p1.startswith('4'): return "Tetragonal"
        if base_p1.startswith('3'): return "Trigonal"
        
        # Orthorhombic: 3 distinct symmetry elements without higher-order axes
        if len(tokens) == 3: return "Orthorhombic"
        
        # Triclinic: Only 1-fold axis (1 or -1)
        if base_p1.startswith('1') and len(tokens)==1: return "Triclinic"
        
        # Default fallback
        return "Monoclinic"

    def _explain_token(self, token):
        """Translates a single symbol token into English text."""
        # Case: Axis with perpendicular plane (e.g., 6/m)
        if '/' in token:
            ax, pl = token.split('/')
            return f"{self._explain_axis(ax)} perpendicular to a {self._explain_plane(pl)}"
        # Case: Axis only
        if any(c.isdigit() for c in token): return self._explain_axis(token)
        # Case: Plane only
        return f"a {self._explain_plane(token)}"

    def _explain_axis(self, t):
        if '_' in t: return f"a {t}-screw axis"
        if '-' in t: return f"a {t.replace('-','')}-fold rotoinversion axis"
        return f"a {t}-fold rotation axis"

    def _explain_plane(self, t):
        return "mirror plane" if t == 'm' else f"{t}-glide plane"

# ==============================================================================
# 2. MatBERT Embedding Generator (768-dim)
#    - Loads pre-trained MatBERT model optimized for materials science text.
# ==============================================================================
class SpaceGroupEmbedder:
    def __init__(self, model_path="data_preprocessing/embedding_space_group.v2/matbert-base-uncased", batch_size=8):
        print(f"🔄 Loading MatBERT model from: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)
        self.model = BertModel.from_pretrained(model_path).to(self.device)
        self.batch_size = batch_size
        print(f"✅ Model loaded on {self.device}")

    def get_embeddings(self, text_list):
        """
        Converts a list of texts into fixed-size embeddings using the CLS token.
        """
        self.model.eval()
        all_embs = []
        
        # Process in batches to manage GPU memory
        for i in tqdm(range(0, len(text_list), self.batch_size), desc="Embedding"):
            batch = text_list[i : i + self.batch_size]
            
            # Tokenization
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, 
                max_length=512, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Extract the [CLS] token embedding (first token of the sequence)
                # Shape: (batch_size, 768)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embs.extend(cls_emb)
                
        return all_embs

# ==============================================================================
# 3. Main Execution Workflow
# ==============================================================================
def main():
    # Define file paths
    # Note: Update 'input_path' to match your local environment
    input_path = r"data_preprocessing\1_MatDX_EF.csv"
    output_json_path = "data_preprocessing\embedding_space_group.v2\space_group_descriptions.json"

    print(f"📂 Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("❌ File not found. Please check the input path.")
        return
    
    # 1. Extract Unique Space Groups
    # We only need to embed unique symbols to save computation time.
    if 'space_group' not in df.columns:
        print("❌ 'space_group' column is missing in the CSV.")
        return

    unique_symbols = df['space_group'].dropna().unique()
    print(f"📊 Found {len(unique_symbols)} unique space groups.")

    # 2. Generate Natural Language Descriptions
    print("📝 Generating natural language descriptions...")
    describer = SpaceGroupDescriber()
    symbol_to_desc = {sym: describer.generate_description(sym) for sym in unique_symbols}
    
    # Preview a sample description
    sample_sym = unique_symbols[0]
    print(f"   [Preview] {sample_sym}: {symbol_to_desc.get(sample_sym, '')[:80]}...")

    # 3. Generate Embeddings using MatBERT
    print("🧠 Generating MatBERT embeddings (768-dim)...")
    embedder = SpaceGroupEmbedder(batch_size=64)
    
    # Ensure order consistency
    symbols_list = list(symbol_to_desc.keys())
    texts_list = [symbol_to_desc[s] for s in symbols_list]
    
    embeddings = embedder.get_embeddings(texts_list)
    
    # 4. Save Description Texts as JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(symbol_to_desc, f, ensure_ascii=False, indent=2)
    print(f"💾 Description texts saved to {output_json_path}")
    
    # 5. Save Embeddings as CSV
    output_csv_path = r"data_preprocessing\embedding_space_group.v2\space_group_embeddings.csv"
    emb_df = pd.DataFrame(embeddings, index=symbols_list)
    emb_df.columns = [f"dim_{i}" for i in range(len(embeddings[0]))]
    emb_df.index.name = "space_group"
    emb_df.to_csv(output_csv_path)
    print(f"💾 Embedding CSV saved to {output_csv_path}")

    print(f"✅ Process Complete. Vector Size: {len(embeddings[0])}")

if __name__ == "__main__":
    main()