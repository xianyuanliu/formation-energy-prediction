# CSV Merger: Combine CIF data with space group embeddings
import pandas as pd
import os

def merge_cif_with_embeddings(
    cif_csv_path="data/cifs/1_MatDX_EF_modified.csv",
    embeddings_csv_path="data_preprocessing/spacegroup_textgen/spacegroup_embeddings_384d.csv",
    output_csv_path="data/SG_text_data.csv"
):
    """
    Merge CIF data with space group embeddings
    
    Args:
        cif_csv_path: Path to CIF data CSV
        embeddings_csv_path: Path to space group embeddings CSV  
        output_csv_path: Output file path
    
    Output CSV structure:
        Composition | space_group | emb_000 | emb_001 | ... | emb_383
    """
    
    print("🔗 Starting CSV merge process...")
    print("=" * 60)
    
    # Step 1: Load CIF data
    try:
        print(f"📁 Loading CIF data from: {cif_csv_path}")
        if not os.path.exists(cif_csv_path):
            raise FileNotFoundError(f"CIF CSV not found: {cif_csv_path}")
        
        cif_df = pd.read_csv(cif_csv_path)
        print(f"✓ Loaded CIF data: {cif_df.shape}")
        print(f"   Columns: {list(cif_df.columns)}")
        
        # Check required columns
        if 'file_name' not in cif_df.columns:
            raise ValueError("'file_name' column not found in CIF CSV")
        if 'space_group' not in cif_df.columns:
            raise ValueError("'space_group' column not found in CIF CSV")
        
        print(f"   Unique space groups in CIF data: {cif_df['space_group'].nunique()}")
        
    except Exception as e:
        print(f"❌ Error loading CIF data: {e}")
        return False
    
    # Step 2: Load embeddings data
    try:
        print(f"\n📁 Loading embeddings from: {embeddings_csv_path}")
        if not os.path.exists(embeddings_csv_path):
            raise FileNotFoundError(f"Embeddings CSV not found: {embeddings_csv_path}")
        
        embeddings_df = pd.read_csv(embeddings_csv_path)
        print(f"✓ Loaded embeddings: {embeddings_df.shape}")
        
        # Check embedding structure
        if 'space_group' not in embeddings_df.columns:
            raise ValueError("'space_group' column not found in embeddings CSV")
        
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('emb_')]
        if len(embedding_cols) == 0:
            raise ValueError("No embedding columns (emb_000, emb_001, ...) found")
        
        print(f"   Space groups in embeddings: {len(embeddings_df)}")
        print(f"   Embedding dimensions: {len(embedding_cols)}")
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Step 3: Prepare CIF data
    print(f"\n🔄 Preparing CIF data...")
    
    # Select and rename columns
    cif_subset = cif_df[['file_name', 'space_group']].copy()
    cif_subset = cif_subset.rename(columns={'file_name': 'Composition'})
    
    print(f"✓ CIF data prepared:")
    print(f"   Rows: {len(cif_subset)}")
    print(f"   Columns: {list(cif_subset.columns)}")
    
    # Step 4: Merge with embeddings
    print(f"\n🔗 Merging with embeddings...")
    
    try:
        # Perform left join to keep all CIF records
        merged_df = pd.merge(
            cif_subset, 
            embeddings_df, 
            on='space_group', 
            how='left'
        )
        
        print(f"✓ Merge completed:")
        print(f"   Output shape: {merged_df.shape}")
        print(f"   Total columns: {len(merged_df.columns)}")
        
        # Check for missing embeddings
        missing_embeddings = merged_df[embedding_cols[0]].isna().sum()
        if missing_embeddings > 0:
            print(f"⚠️  Records with missing embeddings: {missing_embeddings}")
            
            # Show which space groups are missing
            missing_space_groups = merged_df[merged_df[embedding_cols[0]].isna()]['space_group'].unique()
            print(f"   Missing space groups: {list(missing_space_groups)[:10]}")
        else:
            print(f"✓ All records have embeddings!")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        return False
    
    # Step 5: Save output
    print(f"\n💾 Saving merged data...")
    
    try:
        merged_df.to_csv(output_csv_path, index=False)
        
        # Verify saved file
        if os.path.exists(output_csv_path):
            file_size = os.path.getsize(output_csv_path) / 1024 / 1024  # MB
            print(f"✓ Saved successfully: {output_csv_path}")
            print(f"   File size: {file_size:.2f} MB")
        else:
            print(f"❌ File was not created: {output_csv_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False
    
    # Step 6: Show final structure
    print(f"\n📊 Final CSV Structure:")
    print(f"   Shape: {merged_df.shape}")
    print(f"   Columns: Composition + space_group + {len(embedding_cols)} embeddings")
    
    # Show preview
    print(f"\n👀 Preview (first 3 rows, first 6 columns):")
    preview = merged_df.iloc[:3, :6]
    print(preview.to_string(index=False))
    print("   ...")
    
    # Show column structure
    print(f"\n📋 Column Structure:")
    print(f"   1. Composition: {merged_df['Composition'].dtype}")
    print(f"   2. space_group: {merged_df['space_group'].dtype}")
    print(f"   3-{len(embedding_cols)+2}. emb_000 to emb_383: {merged_df[embedding_cols[0]].dtype}")
    
    print(f"\n✅ Merge process completed successfully!")
    return True

def validate_output(csv_path):
    """Validate the output CSV structure"""
    try:
        df = pd.read_csv(csv_path)
        
        print(f"\n🔍 Validating output: {csv_path}")
        print(f"   Shape: {df.shape}")
        
        # Check required columns
        required_cols = ['Composition', 'space_group']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ Missing required column: {col}")
                return False
        
        # Check embedding columns
        embedding_cols = [col for col in df.columns if col.startswith('emb_')]
        if len(embedding_cols) != 384:
            print(f"⚠️  Expected 384 embedding columns, found {len(embedding_cols)}")
        
        # Check for missing values in embeddings
        missing_count = df[embedding_cols].isna().any(axis=1).sum()
        if missing_count > 0:
            print(f"⚠️  {missing_count} rows have missing embedding values")
        
        print(f"✓ Validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 CIF + Space Group Embeddings Merger")
    print("Creates: Composition | space_group | emb_000 | ... | emb_383")
    print()
    
    # File paths
    cif_csv = "data/cifs/1_MatDX_EF_modified.csv"
    embeddings_csv = "data/spacegroup_embeddings_384d.csv" 
    output_csv = "sg_text_data.csv"
    
    # Check input files exist
    missing_files = []
    for file_path in [cif_csv, embeddings_csv]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing input files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Make sure all input files exist before running.")
        return
    
    # Run merge process
    success = merge_cif_with_embeddings(
        cif_csv_path=cif_csv,
        embeddings_csv_path=embeddings_csv,
        output_csv_path=output_csv
    )
    
    if success:
        # Validate output
        validate_output(output_csv)
        
        print(f"\n🎉 Success! Output ready: {output_csv}")
        print(f"📋 Use this file with XRDDataset-style loading:")
        print(f"   - Key column: 'Composition' (renamed from file_name)")
        print(f"   - Space group preserved for reference")
        print(f"   - 384 embedding dimensions: emb_000 to emb_383")
    else:
        print(f"\n❌ Merge failed. Please check error messages above.")

if __name__ == "__main__":
    main()