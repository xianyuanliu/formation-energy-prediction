import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from matminer.featurizers.structure import XRDPowderPattern
from matminer.featurizers.structure.symmetry import GlobalSymmetryFeatures

# --- 1. Configuration --- 🔧
CIF_FOLDER = "CIFs"
PLOT_FOLDER = "XRD_Patterns"
OUTPUT_CSV_FILE = "crystal_features.csv"

# --- 2. Setup Paths and Find Files --- 📁
base_dir = os.path.dirname(os.path.abspath(__file__))
cif_dir = os.path.join(base_dir, CIF_FOLDER)
plot_dir = os.path.join(base_dir, PLOT_FOLDER)

# Create plot directory if it doesn't exist
os.makedirs(plot_dir, exist_ok=True)

cif_files = glob.glob(os.path.join(cif_dir, '*.cif'))

if not cif_files:
    print(f"❌ Error: No CIF files found in the '{cif_dir}' folder. Please check the path.")
    exit()

# --- 3. Initialize Featurizers --- ✨
gsf_featurizer = GlobalSymmetryFeatures()
xrd_vector_featurizer = XRDPowderPattern(two_theta_range=(0, 127), wavelength='CuKa')

# XRD Calculator for generating plots (pymatgen)
xrd_calculator = XRDCalculator(wavelength='CuKa')

# --- 4. Main Processing Loop --- 🚀
all_results = []
print(f"✅ Found {len(cif_files)} CIF files. Starting feature extraction.")

for cif_path in tqdm(cif_files, desc="Processing CIFs"):
    try:
        filename_no_ext = os.path.splitext(os.path.basename(cif_path))[0]
        structure = Structure.from_file(cif_path)
        
        # (1) Calculate features for CSV storage
        result_dict = {'Composition': filename_no_ext}
        
        # Calculate and add symmetry features
        gsf_vector = gsf_featurizer.featurize(structure)
        gsf_labels = gsf_featurizer.feature_labels()
        result_dict.update(zip(gsf_labels, gsf_vector))
        
        # Calculate and add XRD features
        xrd_vector = xrd_vector_featurizer.featurize(structure)
        xrd_labels = xrd_vector_featurizer.feature_labels()
        result_dict.update(zip(xrd_labels, xrd_vector))

        all_results.append(result_dict)
        
        # (2) Generate and save XRD pattern plot using pymatgen
        xrd_pattern = xrd_calculator.get_pattern(structure, two_theta_range=(0, 127))
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(xrd_pattern.x, xrd_pattern.y, 'b-', linewidth=1)
        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(f'XRD Pattern - {filename_no_ext}')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 130)
        
        # Save the plot
        plot_filename = f"{filename_no_ext}.png"
        plt.savefig(os.path.join(plot_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory

    except Exception as e:
        tqdm.write(f"⚠️ An error occurred while processing '{os.path.basename(cif_path)}': {e}")

# --- 5. Save Final Data --- 💾
if all_results:
    df = pd.DataFrame(all_results)
    
    # Move the 'Composition' column to the front
    cols = ['Composition'] + [col for col in df.columns if col != 'Composition']
    df = df[cols]
    
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\n🎉 All tasks complete! {len(all_results)} entries were saved to '{OUTPUT_CSV_FILE}'.")
    print(f"📈 XRD pattern plots have been saved in the '{PLOT_FOLDER}' folder.")
else:
    print("No files were processed, so no CSV file was created.")