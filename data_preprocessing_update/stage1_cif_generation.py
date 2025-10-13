def generate_cifs_from_csv(file_name: str):
 
    import ast
    import math
    import pandas as pd
    import numpy as np
    from collections import Counter
    from pathlib import Path
    from config_path import BASE_DIR

    # --------------------
    def angle_deg(u, v):
        u, v = np.array(u, float), np.array(v, float)
        cosang = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.degrees(math.acos(cosang)))

    def write_cif_file(num_str, formula, space_group, len_a, len_b, len_c,
                       angle_alpha, angle_beta, angle_gamma, volume,
                       site_rows, elements, cif_name, encoding="utf-8"):
        lines = []
        lines.append("# generated using pymatgen")
        lines.append(f"data_{num_str}_{formula}")
        lines.append(f"_symmetry_space_group_name_H-M   '{space_group}'")
        lines.append(f"_cell_length_a   {len_a:.8f}")
        lines.append(f"_cell_length_b   {len_b:.8f}")
        lines.append(f"_cell_length_c   {len_c:.8f}")
        lines.append(f"_cell_angle_alpha   {angle_alpha:.8f}")
        lines.append(f"_cell_angle_beta    {angle_beta:.8f}")
        lines.append(f"_cell_angle_gamma   {angle_gamma:.8f}")
        lines.append(f"_chemical_formula_structural   {formula}")
        lines.append(f"_chemical_formula_sum   '{' '.join(f'{el}{Counter(elements)[el]}' for el in sorted(Counter(elements)))}'")
        lines.append(f"_cell_volume   {volume:.8f}")
        lines.append("loop_")
        lines.append(" _symmetry_equiv_pos_site_id")
        lines.append(" _symmetry_equiv_pos_as_xyz")
        lines.append("  1  'x, y, z'")
        lines.append("loop_")
        lines.append(" _atom_type_symbol")
        for el in sorted(set(elements)):
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
            lines.append(
                f"  {r['type_symbol']}  {r['label']}  {r['mult']}  "
                f"{r['fx']:.8f}  {r['fy']:.8f}  {r['fz']:.8f}  {r['occ']:.0f}"
            )
        text = "\n".join(lines) + "\n"
        cifs_dir = BASE_DIR / "cifs_v1"
        cifs_dir.mkdir(parents=True, exist_ok=True)
        out_path = cifs_dir / cif_name
        out_path.write_text(text, encoding=encoding)
        print(f"Created CIF: {out_path}")

    def write_csv_file(file_path, cif_name, formula, space_group, value_per_atom, value, id_value):
        out_path = BASE_DIR / f"{file_path.stem}_v1.csv"
        row = {
            "cif_name": cif_name,
            "formula": formula,
            "space_group": space_group,
            "value per atom": value_per_atom,
            "value": value,
            "id": id_value,
        }
        df = pd.DataFrame([row])
        if out_path.exists():
            df.to_csv(out_path, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df.to_csv(out_path, index=False, encoding="utf-8")

    def record_skipped_row(num_str, formula):
        out_path = BASE_DIR / f"{file_path.stem}_v1_skipped_rows.csv"
        row = {"num": int(num_str),
               "formula": str(formula) if pd.notna(formula) else "",
               "reason": "insufficient structure data"}
        df = pd.DataFrame([row])
        if out_path.exists():
            df.to_csv(out_path, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Skipped row {num_str}")

    def structure_valid_test(structure_str):
        parsed = ast.literal_eval(structure_str)
        if not isinstance(parsed, list) or not parsed:
            return False
        item = parsed[0]
        if not isinstance(item, dict):
            return False
        data = item.get("data", None)
        if not isinstance(data, dict) or not data:
            return False
        if not all(k in data for k in ("a", "b", "c", "atoms")):
            return False
        if not isinstance(data["atoms"], list) or len(data["atoms"]) == 0:
            return False
        return True

    # ---------- main ----------
    file_path = BASE_DIR / file_name
    df = pd.read_csv(file_path)
    nmax = len(df)
    print(f"Total rows in CSV: {nmax}")

    for num in range(1, nmax + 1):
        num_str = f"{int(num):04d}"
        formula = df.loc[num - 1, "formula"]
        space_group = df.loc[num - 1, "space_group"]
        id_value = df.loc[num - 1, "id"]

        formation_energy = ast.literal_eval(df.loc[num - 1, "formation_energy"])
        value_per_atom = float(formation_energy["value_per_atom"])
        value = float(formation_energy["value"])

        structure_str = df.loc[num - 1, "structure"]
        if structure_valid_test(structure_str):
            structure = ast.literal_eval(structure_str)[0]
            a_vec = np.array(structure["data"]["a"], float) / 1e-10
            b_vec = np.array(structure["data"]["b"], float) / 1e-10
            c_vec = np.array(structure["data"]["c"], float) / 1e-10
            len_a, len_b, len_c = map(float, [np.linalg.norm(a_vec), np.linalg.norm(b_vec), np.linalg.norm(c_vec)])
            angle_alpha = angle_deg(b_vec, c_vec)
            angle_beta = angle_deg(a_vec, c_vec)
            angle_gamma = angle_deg(a_vec, b_vec)
            volume = abs(float(np.dot(a_vec, np.cross(b_vec, c_vec))))

            elements, atom_frac = [], []
            for at in structure["data"]["atoms"]:
                rc = np.array([at["x"], at["y"], at["z"]], float) / 1e-10
                f = np.linalg.solve(np.column_stack([a_vec, b_vec, c_vec]), rc) % 1
                elements.append(at["element"])
                atom_frac.append(f)

            cnt = Counter()
            site_rows = []
            for el, f in zip(elements, atom_frac):
                label = f"{el}{cnt[el]}"
                cnt[el] += 1
                site_rows.append({
                    "type_symbol": el, "label": label, "mult": 1,
                    "fx": f[0], "fy": f[1], "fz": f[2], "occ": 1.0
                })

            cif_name = f"{num_str}_{formula}.cif"
            write_cif_file(num_str, formula, space_group, len_a, len_b, len_c,
                           angle_alpha, angle_beta, angle_gamma, volume,
                           site_rows, elements, cif_name)
            write_csv_file(file_path, cif_name, formula, space_group, value_per_atom, value, id_value)

        else:
            record_skipped_row(num_str, formula)

    print("Done")

init_file = "1_MatDX_EF.csv"
generate_cifs_from_csv(init_file)

        