# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
    from dgl.nn.pytorch.glob import AvgPooling as DGLAvgPooling
except ImportError:
    dgl = None

class RBFExpansion(nn.Module):
    """Simple RBF for distances/angles."""
    def __init__(self, vmin: float, vmax: float, bins: int):
        super().__init__()
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        self.gamma = (bins / (vmax - vmin + 1e-9))**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E,) or (T,)
        x = x.unsqueeze(-1)  # (..., 1)
        return torch.exp(-self.gamma * (x - self.centers) ** 2)  # (..., bins)


class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )
    def forward(self, x): return self.layer(x)


class EdgeGatedGraphConv(nn.Module):
    def __init__(self, input_features: int, output_features: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(self, g: "dgl.DGLGraph", node_feats: torch.Tensor, edge_feats: torch.Tensor):
        g = g.local_var()

        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)     # on g
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)    # on lg

    def forward(self, g: "dgl.DGLGraph", lg: "dgl.DGLGraph",
                x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        x, m = self.node_update(g, x, y)
        y, z = self.edge_update(lg, m, z)
        return x, y, z


class ALIGNNBackbone(nn.Module):
    """
    inputs:
      - g: DGLGraph (batched). Need g.ndata["atom_features"], g.edata["r"] 
      - lg: line graph of g(batched). Need lg.ndata["h"] (angles/triplets)
    outputs:
      - graph_emb: (B, hidden_features)
    """
    def __init__(
        self,
        atom_input_features: int = 92,
        edge_input_features: int = 80,
        triplet_input_features: int = 40,
        embedding_features: int = 64,
        hidden_features: int = 256,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
    ):
        super().__init__()
        if dgl is None:
            raise ImportError("ALIGNNBackbone requires DGL. Please `pip install dgl`.")

        self.hidden_features = hidden_features

        # Embedding lays=ers
        self.atom_embedding = MLPLayer(atom_input_features, hidden_features)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0.0, vmax=8.0, bins=edge_input_features),
            MLPLayer(edge_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1.0, vmax=1.0, bins=triplet_input_features),
            MLPLayer(triplet_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        # stack of ALIGNN layers
        self.alignn_layers = nn.ModuleList(
            [ALIGNNConv(hidden_features, hidden_features) for _ in range(alignn_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(hidden_features, hidden_features) for _ in range(gcn_layers)]
        )

        self.readout = DGLAvgPooling()

    @torch.no_grad()
    def _edge_length(self, g: "dgl.DGLGraph") -> torch.Tensor:
        # g.edata["r"] : (E, 3) vector -> length (E,)
        r = g.edata["r"]
        return torch.norm(r, dim=1)

    def forward(self, g: "dgl.DGLGraph", lg: "dgl.DGLGraph") -> torch.Tensor:
        # angle(triplet) features
        lg = lg.local_var()
        z = self.angle_embedding(lg.edata.pop("h"))  # (T, hidden)

        # node features
        g = g.local_var()
        x = g.ndata.pop("atom_features")             # (N, atom_input_features)
        x = self.atom_embedding(x)                   # (N, hidden)

        # edge features
        bondlen = self._edge_length(g)               # (E,)
        y = self.edge_embedding(bondlen)             # (E, hidden)

        # Update ALIGNN (node/edge/triplet)
        for layer in self.alignn_layers:
            x, y, z = layer(g, lg, x, y, z)

        # Update GCN  (node/edge)
        for layer in self.gcn_layers:
            x, y = layer(g, x, y)

        graph_emb = self.readout(g, x)               # (B, hidden)
        return graph_emb

