"""
1) CIF generation: Only create CIF files for rows where the structure has sufficient information
2) CSV processing: Extract only rows with sufficient structure into a new CSV

Applied requests:
- Processed CSV column order: file_name → formula → space_group → value per atom → value → id
- skipped_rows.csv: only record formula and reason
- CIF internal data_ prefix = starts from 1
- file_name = "number_formula.cif" (e.g., 1_Ni4Ta6.cif)
"""

# ======== User Settings ========
INPUT_CSV = "1_MatDX_EF.csv"          # Input CSV
OUT_DIR   = "result"                  # CIF output folder
OUT_CSV   = "1_MatDX_EF_modified.csv" # Processed CSV filename
STOP_ON_EMPTY_FORMULA = True          # Stop CIF loop if formula is empty
# =================================

import os, ast, json, math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ---------- Structure parsing/validation ----------
def parse_structure_field(value: Any) -> List[dict]:
    if isinstance(value, (list, dict)):
        s = value
    elif pd.isna(value):
        raise ValueError("structure is NaN")
    else:
        t = str(value).strip()
        try: s = json.loads(t)
        except Exception: s = ast.literal_eval(t)
    if isinstance(s, dict): s = [s]
    if not isinstance(s, list) or not s:
        raise ValueError("parsed structure is invalid")
    return s

def has_sufficient_structure(row: pd.Series) -> bool:
    try:
        s0 = parse_structure_field(row["structure"])[0]
        if "data" not in s0: return False
        for k in ("a", "b", "c", "atoms"):
            if k not in s0["data"]: return False
        return True
    except Exception:
        return False

def detect_and_to_angstrom(v):
    arr = np.array(v, float)
    return arr/1e-10 if np.linalg.norm(arr) < 1e-6 else arr

def vec_len(v): return float(np.linalg.norm(np.array(v, float)))

def angle_deg(u, v):
    u, v = np.array(u,float), np.array(v,float)
    cosang = float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(math.acos(cosang)))

def cart_to_frac(a_vec, b_vec, c_vec, r_cart):
    M = np.column_stack([a_vec,b_vec,c_vec])
    f = np.linalg.solve(M,r_cart)
    return f - np.floor(f)

def normalize_space_group(sg):
    if not isinstance(sg,str) or not sg: return "P 1"
    return sg.replace("-3"," -3 ").replace(":"," : ").strip()

def make_formula_sum(elements: List[str]) -> str:
    c = Counter(elements)
    return " ".join(f"{el}{c[el]}" for el in sorted(c))


# ---------- CIF generation ----------
def build_cif_text(formula_structural, space_group,
                   a_len,b_len,c_len, alpha,beta,gamma, volume,
                   site_rows, atom_types, data_prefix=""):
    lines = []
    lines.append("# generated using pymatgen")
    lines.append(f"data_{data_prefix}{formula_structural}")
    lines.append(f"_symmetry_space_group_name_H-M   '{space_group}'")
    lines.append(f"_cell_length_a   {a_len:.8f}")
    lines.append(f"_cell_length_b   {b_len:.8f}")
    lines.append(f"_cell_length_c   {c_len:.8f}")
    lines.append(f"_cell_angle_alpha   {alpha:.8f}")
    lines.append(f"_cell_angle_beta    {beta:.8f}")
    lines.append(f"_cell_angle_gamma   {gamma:.8f}")
    lines.append(f"_chemical_formula_structural   {formula_structural}")
    lines.append(f"_chemical_formula_sum   '{make_formula_sum(atom_types)}'")
    lines.append(f"_cell_volume   {volume:.8f}")
    lines.append("loop_")
    lines.append(" _symmetry_equiv_pos_site_id")
    lines.append(" _symmetry_equiv_pos_as_xyz")
    lines.append("  1  'x, y, z'")
    lines.append("loop_")
    lines.append(" _atom_type_symbol")
    for el in sorted(set(atom_types)):
        lines.append(f"  {el}")
    lines.append("loop_")
    lines.append(" _atom_site_type_symbol")
    lines.append(" _atom_site_label")
    lines.append(" _atom_site_symmetry_multiplicity")
    lines.append(" _atom_site_fract_x")
    lines.append(" _atom_site_fract_y")
    lines.append(" _atom_site_fract_z")
    lines.append(" _atom_site_occupancy")
    for r in site_rows:
        lines.append(f"  {r['type_symbol']}  {r['label']}  {r['mult']}  "
                     f"{r['fx']:.8f}  {r['fy']:.8f}  {r['fz']:.8f}  {r['occ']:.0f}")
    return "\n".join(lines)+"\n"

def write_cif_from_row(row: pd.Series, file_number: int, data_prefix_num: int, out_dir: Path) -> Path:
    s0 = parse_structure_field(row["structure"])[0]
    cell = s0["data"]
    a_vec = detect_and_to_angstrom(cell["a"])
    b_vec = detect_and_to_angstrom(cell["b"])
    c_vec = detect_and_to_angstrom(cell["c"])
    a_len,b_len,c_len = vec_len(a_vec),vec_len(b_vec),vec_len(c_vec)
    alpha, beta, gamma = angle_deg(b_vec,c_vec), angle_deg(a_vec,c_vec), angle_deg(a_vec,b_vec)
    volume = float(np.dot(a_vec,np.cross(b_vec,c_vec)))

    elements, atom_frac = [], []
    for at in cell["atoms"]:
        rc = detect_and_to_angstrom([at["x"],at["y"],at["z"]])
        f = cart_to_frac(a_vec,b_vec,c_vec,rc)
        elements.append(at["element"]); atom_frac.append(f)

    cnt = Counter()
    site_rows = []
    for el,f in zip(elements,atom_frac):
        label=f"{el}{cnt[el]}"; cnt[el]+=1
        site_rows.append({"type_symbol":el,"label":label,"mult":1,
                          "fx":f[0],"fy":f[1],"fz":f[2],"occ":1.0})

    formula_structural = (row.get("formula") or "".join(sorted(set(elements)))).strip()
    sg = normalize_space_group(row.get("space_group"))
    text = build_cif_text(formula_structural,sg,a_len,b_len,c_len,
                          alpha,beta,gamma,volume,site_rows,elements,
                          data_prefix=f"{data_prefix_num}_")

    safe_formula = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in formula_structural) or "structure"
    out_path = out_dir / f"{file_number}_{safe_formula}.cif"
    out_path.write_text(text,encoding="utf-8")
    return out_path


# ---------- formation_energy ----------
def fe_to_dict(x: Any) -> Dict[str,Any]:
    if isinstance(x,dict): return x
    if pd.isna(x): return {}
    try: return ast.literal_eval(str(x))
    except Exception: return {}
def get_val(d: Dict[str,Any], key:str):
    try: return d.get(key,None)
    except Exception: return None


# ---------- Main ----------
def run():
    df = pd.read_csv(INPUT_CSV)
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    created, skipped_rows_log = 0, []
    ok_mask = []
    data_prefix_counter, file_counter = 1, 1

    for _, row in df.iterrows():
        formula_str = str(row.get("formula")).strip() if not pd.isna(row.get("formula")) else ""

        if STOP_ON_EMPTY_FORMULA and formula_str == "":
            print(f"Stop at file {file_counter}: empty 'formula'.")
            break

        if not has_sufficient_structure(row):
            skipped_rows_log.append((f"{file_counter}_{formula_str}", "insufficient 'structure'"))
            ok_mask.append(False)
            file_counter += 1
            continue

        try:
            write_cif_from_row(row, file_counter, data_prefix_counter, out_dir)
            data_prefix_counter += 1; created += 1; ok_mask.append(True)
        except Exception as e:
            skipped_rows_log.append((formula_str,str(e)))
            ok_mask.append(False)
        file_counter += 1

    if skipped_rows_log:
        pd.DataFrame(skipped_rows_log,columns=["formula","reason"]).to_csv(out_dir/"skipped_rows.csv",index=False)

    print(f"[CIF] Created: {created}, Skipped: {len(skipped_rows_log)}")

    ok_df = df.iloc[:len(ok_mask)].copy()
    ok_df = ok_df[pd.Series(ok_mask,index=ok_df.index)]
    if len(ok_df)==0:
        pd.DataFrame(columns=["file_name","formula","space_group","value per atom","value","id"]).to_csv(OUT_CSV,index=False)
        return

    fe_series = ok_df["formation_energy"].apply(fe_to_dict) if "formation_energy" in ok_df.columns \
                else pd.Series([{}]*len(ok_df),index=ok_df.index)
    # number_formula format
    file_names = [f"{i+1}_{str(f).strip()}.cif" for i,f in enumerate(ok_df["formula"])]

    processed = pd.DataFrame({
        "file_name": file_names,
        "formula": ok_df.get("formula",pd.Series([""]*len(ok_df),index=ok_df.index)),
        "space_group": ok_df.get("space_group", pd.Series([None]*len(ok_df), index=ok_df.index)),
        "value per atom": [get_val(d,"value_per_atom") for d in fe_series],
        "value": [get_val(d,"value") for d in fe_series],
        "id": ok_df.get("id",pd.Series([None]*len(ok_df),index=ok_df.index)),
    })[["file_name","formula","space_group","value per atom","value","id"]]

    processed.to_csv(OUT_CSV,index=False)
    print(f"[CSV] Wrote: {OUT_CSV}")


if __name__=="__main__":
    run()
