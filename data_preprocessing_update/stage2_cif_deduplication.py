import pandas as pd
from pathlib import Path
import shutil
from config_path import BASE_DIR

def filtering_same_f_same_sg(file_name, a_file, drop_single = True, value_col = "value per atom", cif_col = "cif_name"):
    file_name = Path(file_name) 
    a_file = BASE_DIR / a_file

    df = pd.read_csv(a_file)
    def pick_median_filename(group: pd.DataFrame) -> str:
        m = group[value_col].median()
        diffs = (group[value_col] - m).abs()
        min_diff = diffs.min()
        candidates = group.loc[diffs == min_diff, [cif_col, value_col]]
        return candidates.sort_values([cif_col]).iloc[0][cif_col]

    def agg_func(group: pd.DataFrame) -> pd.Series:
        cnt = len(group)
        median_val = group[value_col].median()
        median_fname = pick_median_filename(group)
        other_fnames = ";".join(
            sorted(f for f in group[cif_col].dropna().unique() if f != median_fname)
        )
        return pd.Series({
            "count": cnt,
            "median": median_val,
            "median_cif_name": median_fname,
            "other_cif_names": other_fnames
        })

    result = (
        df.groupby(["formula", "space_group"], dropna=False)
          .apply(agg_func)
          .reset_index()
          .sort_values(["count", "formula", "space_group"], ascending=[False, True, True])
    )

    if drop_single:
        result = result[result["count"] > 1].copy()

    result = result[[
        "formula", "space_group", "count",
        "median", "median_cif_name", "other_cif_names"
    ]]
    out_path = BASE_DIR / f"{file_name.stem}_v2_detail.csv"
    result.to_csv(out_path, index=False)

    to_delete = {
        fname.strip()
        for names in result["other_cif_names"].dropna().tolist()
        for fname in (names.split(";") if isinstance(names, str) else [])
        if fname.strip()
    }

    df_v2 = df[~df[cif_col].astype(str).isin(to_delete)].copy()
    out_v2 = BASE_DIR / f"{file_name.stem}_v2.csv"
    df_v2.to_csv(out_v2, index=False)

    src_dir = BASE_DIR / "cifs_v1"
    dst_dir = BASE_DIR / "cifs_v2"
    if not src_dir.exists():
        print(f"No source folder: {src_dir}")
        return result
    shutil.copytree(src_dir, dst_dir)
    removed = 0
    for fname in to_delete:
        p = dst_dir / fname
        if p.exists():
            p.unlink()      
            removed += 1
    print(f"Deleted {removed} files")
    return result


init_file = "1_MatDX_EF.csv"
a_file = f"{Path(init_file).stem}_v1.csv"
filtering_same_f_same_sg(init_file, a_file)