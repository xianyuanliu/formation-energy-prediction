# csv_to_cif_class.py
import os, ast, json, math
from collections import Counter
from typing import List, Tuple
import numpy as np
import pandas as pd


class CsvToCifConverter:
    """
    A class to convert structural data stored in CSV format into CIF files.

    Rules:
    - Row numbering starts from 2 (i.e., row 2 in the CSV gets file number 2).
    - If the 'formula' column is empty, the conversion stops immediately.
    - Failed rows are recorded in a separate CSV file: skipped_rows.csv.
    """

    def __init__(self, csv_path: str, out_dir: str = "result"):
        self.csv_path = csv_path
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # ---------- low-level helpers ----------
    @staticmethod
    def vec_len(v): 
        """Return the vector length (norm)."""
        return float(np.linalg.norm(np.array(v, float)))

    @staticmethod
    def angle_deg(u, v):
        """Return the angle (in degrees) between two vectors u and v."""
        u = np.array(u, float); v = np.array(v, float)
        cosang = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.degrees(math.acos(cosang)))

    @staticmethod
    def cart_to_frac(a_vec, b_vec, c_vec, r_cart):
        """
        Convert Cartesian coordinates into fractional coordinates with respect to
        lattice vectors a, b, c.
        """
        M = np.column_stack([a_vec, b_vec, c_vec])
        f = np.linalg.solve(M, r_cart)
        return f - np.floor(f)  # wrap into [0,1)

    @staticmethod
    def normalize_space_group(sg):
        """Normalize space group string, fallback to 'P 1' if invalid."""
        if not sg or not isinstance(sg, str): 
            return "P 1"
        return sg.replace("-3", " -3 ").replace(":", " : ").strip()

    @staticmethod
    def make_formula_sum(elements):
        """Generate chemical formula sum like 'Ni4 Ta6' from element list."""
        c = Counter(elements)
        return " ".join(f"{el}{c[el]}" for el in sorted(c))

    @staticmethod
    def parse_structure_field(value):
        """
        Parse the 'structure' column: 
        Try JSON → ast.literal_eval → dict/list.
        Return a list of dictionaries representing structures.
        """
        if isinstance(value, (list, dict)):
            s = value
        elif pd.isna(value):
            raise ValueError("structure is NaN")
        else:
            t = str(value).strip()
            try:
                s = json.loads(t)
            except Exception:
                try:
                    s = ast.literal_eval(t)
                except Exception as e:
                    raise ValueError(f"cannot parse structure: {e}")
        if isinstance(s, dict):
            s = [s]
        if not isinstance(s, list) or not s:
            raise ValueError("parsed structure is neither a list nor a non-empty list")
        return s

    @staticmethod
    def detect_and_to_angstrom(vec):
        """
        Convert a lattice vector to Ångström.
        If the norm is very small (< 1e-6), assume input is in meters and convert.
        """
        v = np.array(vec, float)
        return v/1e-10 if np.linalg.norm(v) < 1e-6 else v

    @staticmethod
    def detect_pos_and_to_angstrom(pos):
        """
        Convert atomic position to Ångström.
        If the norm is very small (< 1e-6), assume input is in meters and convert.
        """
        r = np.array(pos, float)
        return r/1e-10 if np.linalg.norm(r) < 1e-6 else r

    # ---------- CIF builders ----------
    def build_cif_text(self, formula_structural, space_group,
                       a_len, b_len, c_len, alpha, beta, gamma, volume,
                       site_rows, atom_types, data_prefix=""):
        """
        Build a CIF text string from lattice parameters, site information, and metadata.
        """
        lines = []
        lines.append("# generated using pymatgen")
        lines.append(f"data_{data_prefix}{formula_structural}")
        lines.append(f"_symmetry_space_group_name_H-M   '{space_group}'")
        lines.append(f"_cell_length_a   {a_len:.8f}")
        lines.append(f"_cell_length_b   {b_len:.8f}")
        lines.append(f"_cell_length_c   {c_len:.8f}")
        lines.append(f"_cell_angle_alpha   {alpha:.8f}")
        lines.append(f"_cell_angle_beta   {beta:.8f}")
        lines.append(f"_cell_angle_gamma   {gamma:.8f}")
        lines.append(f"_chemical_formula_structural   {formula_structural}")
        lines.append(f"_chemical_formula_sum   '{self.make_formula_sum(atom_types)}'")
        lines.append(f"_cell_volume   {volume:.8f}")

        # Symmetry equivalent positions (default: identity)
        lines.append("loop_")
        lines.append(" _symmetry_equiv_pos_site_id")
        lines.append(" _symmetry_equiv_pos_as_xyz")
        lines.append("  1  'x, y, z'")

        # Atom types
        lines.append("loop_")
        lines.append(" _atom_type_symbol")
        for el in sorted(set(atom_types)):
            lines.append(f"  {el}")

        # Atom site information
        lines.append("loop_")
        lines.append(" _atom_site_type_symbol")
        lines.append(" _atom_site_label")
        lines.append(" _atom_site_symmetry_multiplicity")
        lines.append(" _atom_site_fract_x")
        lines.append(" _atom_site_fract_y")
        lines.append(" _atom_site_fract_z")
        lines.append(" _atom_site_occupancy")
        for r in site_rows:
            lines.append(
                f"  {r['type_symbol']}  {r['label']}  {r['mult']}  "
                f"{r['fx']:.8f}  {r['fy']:.8f}  {r['fz']:.8f}  {r['occ']:.0f}"
            )
        return "\n".join(lines) + "\n"

    def write_cif_from_row(self, row, number: int) -> str:
        """
        Create a CIF file from a single CSV row.
        """
        s0 = self.parse_structure_field(row["structure"])[0]
        if "data" not in s0:
            raise ValueError("structure[0] has no 'data'")
        cell = s0["data"]
        for k in ("a", "b", "c", "atoms"):
            if k not in cell:
                raise ValueError(f"cell missing '{k}'")

        atoms = cell["atoms"]

        # Lattice parameters
        a_vec = self.detect_and_to_angstrom(cell["a"])
        b_vec = self.detect_and_to_angstrom(cell["b"])
        c_vec = self.detect_and_to_angstrom(cell["c"])
        a_len, b_len, c_len = self.vec_len(a_vec), self.vec_len(b_vec), self.vec_len(c_vec)
        alpha = self.angle_deg(b_vec, c_vec)
        beta  = self.angle_deg(a_vec, c_vec)
        gamma = self.angle_deg(a_vec, b_vec)
        volume = float(np.dot(a_vec, np.cross(b_vec, c_vec)))

        # Atomic positions: Cartesian → fractional
        elements, atom_cart = [], []
        for at in atoms:
            if not all(k in at for k in ("x", "y", "z", "element")):
                raise ValueError("atom missing one of x,y,z,element")
            elements.append(at["element"])
            atom_cart.append(self.detect_pos_and_to_angstrom([at["x"], at["y"], at["z"]]))
        atom_frac = [self.cart_to_frac(a_vec, b_vec, c_vec, rc) for rc in atom_cart]

        # Assign site labels
        cnt = Counter(); site_rows = []
        for el, f in zip(elements, atom_frac):
            label = f"{el}{cnt[el]}"; cnt[el] += 1
            site_rows.append({"type_symbol": el, "label": label, "mult": 1,
                              "fx": f[0], "fy": f[1], "fz": f[2], "occ": 1.0})

        formula_structural = (row.get("formula") or "".join(sorted(set(elements)))).strip()
        sg = self.normalize_space_group(row.get("space_group"))

        cif_text = self.build_cif_text(
            formula_structural, sg, a_len, b_len, c_len, alpha, beta, gamma, volume,
            site_rows, elements, data_prefix=f"{number}_"
        )

        safe_formula = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in formula_structural) or "structure"
        out_path = os.path.join(self.out_dir, f"{number}_{safe_formula}.cif")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cif_text)
        return out_path

    # ---------- public API ----------
    def convert(self) -> Tuple[List[str], List[tuple]]:
        """
        Main conversion routine: CSV → multiple CIF files.

        Returns:
            out_files: list of paths to successfully created CIF files
            skipped: list of tuples (row_number(2-based), df_index, reason) for failures
        """
        df = pd.read_csv(self.csv_path)
        out_files: List[str] = []
        skipped: List[tuple] = []

        # Start numbering from 2 → row 2 gets number 2
        for row_number, (df_index, row) in enumerate(df.iterrows(), start=2):
            # If formula is empty, stop immediately (do not record as skipped)
            formula_val = row.get("formula")
            if pd.isna(formula_val) or str(formula_val).strip() == "":
                print(f"Stop at row {row_number}: empty 'formula'.")
                break

            try:
                if "structure" not in row or pd.isna(row["structure"]):
                    raise ValueError("structure is NaN")
                out_files.append(self.write_cif_from_row(row, number=row_number))
            except Exception as e:
                skipped.append((row_number, df_index, str(e)))
                # continue with the next row

        # Save skipped rows into a CSV file
        if skipped:
            pd.DataFrame(skipped, columns=["row_number(2-based)", "df_index", "reason"]) \
              .to_csv(os.path.join(self.out_dir, "skipped_rows.csv"), index=False)

        print(f"Done. Created {len(out_files)} CIFs. Skipped {len(skipped)} rows.")
        return out_files, skipped


if __name__ == "__main__":
    # Example usage
    converter = CsvToCifConverter(csv_path="1_MatDX_EF.csv", out_dir="result")
    files, skipped = converter.convert()
    for p in files:
        print(" -", p)
