import os
import warnings
import numpy as np
import torch
import dgl
import dgl.function as fn
from pymatgen.core import Structure

def _frac_disp_wrap(frac_from, frac_to):
    d = frac_to - frac_from
    d -= np.round(d)
    return d  # fractional displacement

def build_dgl_graphs_from_structure(crystal: Structure, radius: float, max_num_nbr: int):
    """
    CIF -> (g, lg)
    g.ndata['atom_features'] : set in __getitem__ of Dataset
    g.edata['r'] : (E,3) cartesian vector (src->dst)
    lg.edata['h'] : (T,) cos(theta)  (backtracking=False)
    """
    N = len(crystal)
    lattice = crystal.lattice

    cart = np.array([site.coords for site in crystal], dtype=np.float64)
    frac = np.array([site.frac_coords for site in crystal], dtype=np.float64)

    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    src_list, dst_list, r_list = [], [], []
    for i, nbrs in enumerate(all_nbrs):
        if len(nbrs) < max_num_nbr:
            warnings.warn(f'{crystal.composition.reduced_formula} not enough neighbors. Consider increasing radius.')
        nbrs = nbrs[:max_num_nbr]  # truncate
        for site_j, dist_ij, j in nbrs:
            d_frac = _frac_disp_wrap(frac[i], site_j.frac_coords)
            d_cart = lattice.get_cartesian_coords(d_frac)  # vec i->j
            # i->j
            src_list.append(i)
            dst_list.append(j)
            r_list.append(d_cart)
            # j->i 
            src_list.append(j)
            dst_list.append(i)
            r_list.append(-d_cart)

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    r   = torch.tensor(np.vstack(r_list), dtype=torch.float32)  # (E,3)

    g = dgl.graph((src, dst), num_nodes=N)
    g.edata["r"] = r  # atom feature is set in Dataset __getitem__

    E = src.numel()
    in_edges_of = [[] for _ in range(N)]
    out_edges_of = [[] for _ in range(N)]
    for e in range(E):
        s, d = src[e].item(), dst[e].item()
        out_edges_of[s].append(e)
        in_edges_of[d].append(e)

    lg_src, lg_dst, cos_list = [], [], []
    r_unit = F.normalize(r, dim=1)  # (E,3)
    for i in range(N):
        ins = in_edges_of[i]     # j->i
        outs = out_edges_of[i]   # i->k
        if not ins or not outs:
            continue
        for e_in in ins:
            # r_in: j->i
            u_in = -r_unit[e_in]  # (3,)
            src_of_in = src[e_in].item()
            for e_out in outs:
                dst_of_out = dst[e_out].item()
                if src_of_in == dst_of_out:   # No backtracking 
                    continue
                u_out = r_unit[e_out]
                cos_theta = torch.clamp(torch.dot(u_in, u_out), -1.0, 1.0)
                lg_src.append(e_in)
                lg_dst.append(e_out)
                cos_list.append(cos_theta.item())

    if len(lg_src) == 0:
        # handle case with no edges in line graph
        lg_src = [0]; lg_dst = [0]; cos_list = [1.0]

    lg = dgl.graph(
        (torch.tensor(lg_src, dtype=torch.long), torch.tensor(lg_dst, dtype=torch.long)),
        num_nodes=E
    )
    lg.edata["h"] = torch.tensor(cos_list, dtype=torch.float32).unsqueeze(-1)  # (T,1)

    return g, lg
