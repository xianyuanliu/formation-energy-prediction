import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path 
from config_path import BASE_DIR 


def same_f_diff_sg(file_name, b_file):

    import shutil

    file_name = Path(file_name) 
    b_file = BASE_DIR / b_file
    df = pd.read_csv(b_file)

    # 1) setting
    result = (
            df.groupby("formula")["space_group"]
        .agg(
            distinct_space_groups=lambda s: sorted(s.astype(str).unique()),
            n_distinct_space_groups=lambda s: s.astype(str).nunique()
        )
        .reset_index()
        .sort_values(["n_distinct_space_groups", "formula"], ascending=[False, True])
    )
    result = result[["formula", "n_distinct_space_groups", "distinct_space_groups"]]
    
    # 2) make csv
    out_path = BASE_DIR / f"{file_name.stem}_v3_same_f_diff_sg.csv"
    result.to_csv(out_path, index=False)

    # 3) plot bar graph
    max_sg = int(result["n_distinct_space_groups"].max())
    formulas = result["formula"].dropna().astype(str)
    sg_buckets = {k: set(formulas[result["n_distinct_space_groups"] == k].unique()) for k in range(1, max_sg + 1)}

    labels = [str(k) for k in range(1, max_sg + 1)]
    counts = [len(sg_buckets[k]) for k in range(1, max_sg + 1)]
    plt.figure()
    bars = plt.bar(labels, counts)
    plt.xlabel("n_distinct_space_groups")
    plt.ylabel("# of formulas")
    plt.title(f"Formula counts by # of distinct space groups (max={max_sg})")
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, str(count),
                ha='center', va='bottom', fontsize=9)
    fig_path = BASE_DIR / "v3_same_f_diff_sg.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    # 4) make dataset folder
    v3_root = BASE_DIR / "cifs_v3_same_f_diff_sg"
    one_sg_dir = v3_root / "1 sg"
    two_plus_dir = v3_root / "2_plus_sg"
    one_sg_dir.mkdir(parents=True, exist_ok=True)
    two_plus_dir.mkdir(parents=True, exist_ok=True) 
    src_cifs_v2 = BASE_DIR / "cifs_v2"
    dst_cifs_v2 = v3_root / "cifs_v2"
    if not src_cifs_v2.exists():
        print(f"No source folder: {src_cifs_v2} (skip copy & move)")
        return result
    shutil.copytree(src_cifs_v2, dst_cifs_v2)

    one_sg_formulas = sg_buckets.get(1, set())
    moved_one, moved_two = 0, 0
    for p in dst_cifs_v2.glob("*.cif"):
        name = p.name
        try:
            formula_in_name = name.split("_", 1)[1][:-4] 
        except IndexError:
            shutil.move(str(p), two_plus_dir / name)
            moved_two += 1
            continue

        if formula_in_name in one_sg_formulas:
            shutil.move(str(p), one_sg_dir / name)
            moved_one += 1
        else:
            shutil.move(str(p), two_plus_dir / name)
            moved_two += 1
    print(f"Moved to '1 sg': {moved_one} files")
    print(f"Moved to '2_plus_sg': {moved_two} files")
    if not any(dst_cifs_v2.iterdir()):
        shutil.rmtree(dst_cifs_v2)

def plot_periodic_table_from_csv(file_name, b_file):

    import re
    import numpy as np
    from collections import Counter
    from matplotlib import cm, colors
    from matplotlib.patches import Rectangle

    # 1) path
    file_name = Path(file_name) 
    b_file = BASE_DIR / b_file

    # 2) count element_frequencies, make csv
    def ensure_element_stats(out_path, csv_path):
        df_src = pd.read_csv(csv_path)
        element_pattern = re.compile(r"([A-Z][a-z]?)")
        def parse_elements(formula):
            if pd.isna(formula):
                return []
            return element_pattern.findall(str(formula))
        all_elements = []
        for f in df_src["formula"]:
            all_elements.extend(parse_elements(f))
        counter = Counter(all_elements)

        overall_df = (
            pd.DataFrame(counter.items(), columns=["element", "frequency"])
            .sort_values("frequency", ascending=False)
            .reset_index(drop=True)
        )
        overall_df.to_csv(out_path, index=False)

    out_path = BASE_DIR / f"{file_name.stem}_v3_element_freq.csv"
    ensure_element_stats(out_path, b_file)

    # 3) periodic table setting
    df = pd.read_csv(out_path)
    total = float(df["frequency"].sum()) if df.shape[0] else 0.0
    freq = {str(e): float(c) / total * 100.0 
            for e, c in zip(df["element"], df["frequency"]) if isinstance(e, str)}

    ptable_pos = {
        "H":(1,1), "He":(1,18),
        "Li":(2,1),"Be":(2,2),"B":(2,13),"C":(2,14),"N":(2,15),"O":(2,16),"F":(2,17),"Ne":(2,18),
        "Na":(3,1),"Mg":(3,2),"Al":(3,13),"Si":(3,14),"P":(3,15),"S":(3,16),"Cl":(3,17),"Ar":(3,18),
        "K":(4,1),"Ca":(4,2),"Sc":(4,3),"Ti":(4,4),"V":(4,5),"Cr":(4,6),"Mn":(4,7),"Fe":(4,8),
        "Co":(4,9),"Ni":(4,10),"Cu":(4,11),"Zn":(4,12),"Ga":(4,13),"Ge":(4,14),"As":(4,15),"Se":(4,16),"Br":(4,17),"Kr":(4,18),
        "Rb":(5,1),"Sr":(5,2),"Y":(5,3),"Zr":(5,4),"Nb":(5,5),"Mo":(5,6),"Tc":(5,7),"Ru":(5,8),
        "Rh":(5,9),"Pd":(5,10),"Ag":(5,11),"Cd":(5,12),"In":(5,13),"Sn":(5,14),"Sb":(5,15),"Te":(5,16),"I":(5,17),"Xe":(5,18),
        "Cs":(6,1),"Ba":(6,2),"La":(6,3),"Hf":(6,4),"Ta":(6,5),"W":(6,6),"Re":(6,7),"Os":(6,8),
        "Ir":(6,9),"Pt":(6,10),"Au":(6,11),"Hg":(6,12),"Tl":(6,13),"Pb":(6,14),"Bi":(6,15),"Po":(6,16),"At":(6,17),"Rn":(6,18),
        "Fr":(7,1),"Ra":(7,2),"Ac":(7,3),"Rf":(7,4),"Db":(7,5),"Sg":(7,6),"Bh":(7,7),"Hs":(7,8),
        "Mt":(7,9),"Ds":(7,10),"Rg":(7,11),"Cn":(7,12),"Nh":(7,13),"Fl":(7,14),"Mc":(7,15),"Lv":(7,16),"Ts":(7,17),"Og":(7,18),
        "Ce":(8,4),"Pr":(8,5),"Nd":(8,6),"Pm":(8,7),"Sm":(8,8),"Eu":(8,9),"Gd":(8,10),"Tb":(8,11),"Dy":(8,12),
        "Ho":(8,13),"Er":(8,14),"Tm":(8,15),"Yb":(8,16),"Lu":(8,17),
        "Th":(9,4),"Pa":(9,5),"U":(9,6),"Np":(9,7),"Pu":(9,8),"Am":(9,9),"Cm":(9,10),"Bk":(9,11),"Cf":(9,12),
        "Es":(9,13),"Fm":(9,14),"Md":(9,15),"No":(9,16),"Lr":(9,17),
    }
    max_row = max(r for (r, c) in ptable_pos.values())
    max_col = max(c for (r, c) in ptable_pos.values())

    # 4) Transition metal point
    def get_transition_metals(ptable_pos):
        tm = set()
        for el, (r, c) in ptable_pos.items():
            if r <= 7 and 3 <= c <= 12:
                tm.add(el)
        return tm
    transition_metals = get_transition_metals(ptable_pos)

    # 5) plot
    vals = np.array([freq.get(el, np.nan) for el in ptable_pos.keys()])
    finite_vals = vals[np.isfinite(vals)]
    vmin, vmax = (float(np.min(finite_vals)), float(np.max(finite_vals))) if finite_vals.size else (0.0, 1.0)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    base = cm.get_cmap("magma", 256)
    cmap = colors.LinearSegmentedColormap.from_list("lighter_magma", base(np.linspace(0.5, 1, 256)))

    fig, ax = plt.subplots(figsize=(18, 8.5))
    cell_w, cell_h, pad = 1.0, 1.0, 0.06

    for el, (r, c) in ptable_pos.items():
        x = c - 1
        y = max_row - r  
        v = freq.get(el, np.nan)
        face = cmap(norm(v)) if np.isfinite(v) else (0.92, 0.92, 0.92, 1.0)

        edge = 'black'
        lw = 0.5
        if el in transition_metals:
            edge = 'tab:blue'
            lw = 2.0

        rect = Rectangle((x+pad, y+pad), cell_w-2*pad, cell_h-2*pad,
                         linewidth=lw, edgecolor=edge, facecolor=face)
        ax.add_patch(rect)

        ax.text(x + 0.48, y + 0.62, el, fontsize=20, fontweight='bold', va='center', ha='center')
        if np.isfinite(v):
            ax.text(x + 0.2, y + 0.25, f"{v:.1f}%", fontsize=15, va='center', ha='left')

    ax.set_xlim(0, max_col)
    ax.set_ylim(0, max_row)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title("Element Frequency Heatmap (periodic table)", fontsize=25, pad=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Frequency (%)", fontsize=20)

    fig.tight_layout()
    out_png = BASE_DIR / "v3_periodic_table_frequency_heatmap.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def sort_cifs_by_transition_metals(b_file):

    import re
    import shutil 

    # 1) path
    b_file = BASE_DIR / b_file
    root_dir = BASE_DIR / "cifs_v3_transition_metal"
    w_tm_dir = root_dir / "w_TM"
    wo_tm_dir = root_dir / "wo_TM"
    src_cifs_v2 = BASE_DIR / "cifs_v2"
    dst_cifs_v2 = root_dir / "cifs_v2"
    w_tm_dir.mkdir(parents=True, exist_ok=True)
    wo_tm_dir.mkdir(parents=True, exist_ok=True)
    if not src_cifs_v2.exists():
        print(f"[TM sort] No source folder: {src_cifs_v2}")
        return
    if dst_cifs_v2.exists():
        shutil.rmtree(dst_cifs_v2)
    shutil.copytree(src_cifs_v2, dst_cifs_v2)

    # 2) count TM
    transition_metals = {
        "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
        "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
        "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
        "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn"
    }
    df = pd.read_csv(b_file)
    el_pattern = re.compile(r"([A-Z][a-z]?)")

    def tm_count(formula):
        if pd.isna(formula):
            return 0
        elems = el_pattern.findall(str(formula))
        return sum(e in transition_metals for e in elems)
    
    df["TM_count"] = df["formula"].apply(tm_count)
    tm_dict = df.set_index("formula")["TM_count"].to_dict()

    # 3) make dataset folder 
    moved_w, moved_wo, unknown = 0, 0, 0
    for p in dst_cifs_v2.glob("*.cif"):
        formula_in_name = p.name.split("_", 1)[1][:-4]
        tm_val = tm_dict[formula_in_name]
        if tm_val == 0:
            shutil.move(str(p), wo_tm_dir / p.name)
            moved_wo += 1
        elif tm_val in (1, 2):
            shutil.move(str(p), w_tm_dir / p.name)
            moved_w += 1
    if not any(dst_cifs_v2.iterdir()):
        shutil.rmtree(dst_cifs_v2)

    print(f"[TM sort] Moved to 'w_TM'  : {moved_w} files")
    print(f"[TM sort] Moved to 'wo_TM' : {moved_wo} files")

    #4) draw pie chart
    total = moved_w + moved_wo
    if total > 0:
        labels = ["w/ TM", "w/o TM"]
        sizes = [moved_w, moved_wo]
        colors = ["#708993","#D3E6E1" ]
        plt.figure(figsize=(5.5, 5.5))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors, 
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            startangle=90,
            counterclock=False
        )
        plt.title("TM-based CIF distribution")
        plt.axis("equal")  
        pie_path = BASE_DIR / "v3_transition_metal.png"
        plt.savefig(pie_path, dpi=300, bbox_inches="tight")
        plt.close()


def classify_compounds(file_name, b_file):

    import re
    import shutil

    # 1) path
    file_name = Path(file_name) 
    b_file = BASE_DIR / b_file
    root_dir = BASE_DIR / "cifs_v3_general_subdomain"
    src_cifs_v2 = BASE_DIR / "cifs_v2"
    dst_cifs_v2 = root_dir / "cifs_v2"
    domains = ["oxides","nitrides","carbides","halides","chalcogenides","intermetallics","others"]
    domain_dirs = {d: (root_dir / d) for d in domains}
    root_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_cifs_v2, dst_cifs_v2)
    for dpath in domain_dirs.values():
        dpath.mkdir(parents=True, exist_ok=True)
    
    # 2) setting
    df = pd.read_csv(b_file)
    el_pattern = re.compile(r"([A-Z][a-z]?)")

    METALS = {
        "Li","Na","K","Rb","Cs","Fr",
        "Be","Mg","Ca","Sr","Ba","Ra",
        "Sc","Y","Ti","Zr","Hf","V","Nb","Ta","Cr","Mo","W","Mn","Tc","Re",
        "Fe","Ru","Os","Co","Rh","Ir","Ni","Pd","Pt","Cu","Ag","Au","Zn","Cd","Hg",
        "Al","Ga","In","Tl","Sn","Pb","Bi","Po"
    }

    def only_metals(elems_set):
        return len(elems_set) > 0 and all(e in METALS for e in elems_set)
    
    def classify(formula):
        if pd.isna(formula):
            return "others"
        elems = el_pattern.findall(str(formula))
        elems_set = set(elems)
        if "O" in elems_set:
            return "oxides"
        elif "N" in elems_set:
            return "nitrides"
        elif "C" in elems_set:
            return "carbides"
        elif any(x in elems_set for x in ["F", "Cl", "Br", "I"]):
            return "halides"
        elif any(x in elems_set for x in ["S", "Se", "Te"]):
            return "chalcogenides"
        elif only_metals(elems_set):
            return "intermetallics"
        else:
            return "others"

    df["subdomain"] = df["formula"].apply(classify)
    f2d = df.set_index("formula")["subdomain"].to_dict()
 
   # 3) make dataset folder
    moved_counts = {d: 0 for d in domains}
    cif_files = list(dst_cifs_v2.glob("*.cif"))
    for p in cif_files:
        formula_in_name = p.name.split("_", 1)[1][:-4]  # "_” 이후부터 ".cif" 앞까지
        domain = f2d.get(formula_in_name, "others")
        dest = domain_dirs[domain] / p.name
        shutil.move(str(p), str(dest))
        moved_counts[domain] += 1

    for sub in root_dir.iterdir():
        if sub.is_dir() and not any(sub.iterdir()):
            shutil.rmtree(sub)

    for d in domains:
        print(f"  - {d:15s}: {moved_counts[d]}")

    # 4) draw pie chart
    total_moved = sum(moved_counts.values())
    if total_moved > 0:
        sorted_domains = sorted(
            [(d, moved_counts[d]) for d in moved_counts if d != "others"],
            key=lambda x: x[1],
            reverse=True
        )
        if moved_counts["others"] > 0:
            sorted_domains.append(("others", moved_counts["others"]))
        filtered_domains = [(d, c) for d, c in sorted_domains if c > 0]
        labels = [d for d, _ in filtered_domains]
        sizes = [c for _, c in filtered_domains]
        palette = ["#A1C2BD","#B99DDB","#5C5C5C","#9C27B0","#FF5722","#607D8B","#9E9E9E"]
        colors  = palette[:len(labels)]

        plt.figure(figsize=(6.2, 6.2))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            startangle=90,
            counterclock=False,
            textprops={"fontsize": 11}
        )
        plt.title("CIF Distribution by Subdomain", fontsize=13)
        plt.gca().set_aspect("equal")
        pie_path = BASE_DIR / "v3_general_subdomain.png"
        plt.savefig(pie_path, dpi=300, bbox_inches="tight")
        plt.close()
  


# ------ main ---------
init_file = "1_MatDX_EF.csv"
b_file = f"{Path(init_file).stem}_v2.csv"

same_f_diff_sg(init_file, b_file)
plot_periodic_table_from_csv(init_file, b_file)
sort_cifs_by_transition_metals(b_file)
classify_compounds(init_file, b_file)    