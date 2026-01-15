from __future__ import print_function, division, annotations

import torch
import torch.nn as nn
from models.sg_text_module import TextFeatureExtractor
from models.xrd_module import XRDFeatureExtractor

#for CHGNet
from pymatgen.core import Structure
from typing import Optional, Sequence
import warnings

import logging
from typing import TYPE_CHECKING, Literal
from dgl import readout_edges, readout_nodes
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from matgl.layers import (
    ActivationFunction,
    CHGNetAtomGraphBlock,
    CHGNetBondGraphBlock,
    FourierExpansion,
    GatedMLP_norm,
    MLP_norm,
    RadialBesselFunction,
)
from matgl.utils.cutoff import polynomial_cutoff
from matgl.models._core import MatGLModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    import dgl
    from matgl.graph.converters import GraphConverter
#for M3GNet    
from matgl.models._m3gnet import M3GNet

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class MPNNLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, hidden=128, use_bn=True):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.mlp = nn.Sequential(
            nn.Linear(2*atom_fea_len + nbr_fea_len, hidden),
            nn.Softplus(),
            nn.Linear(hidden, atom_fea_len)
        )
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(atom_fea_len)
        self.act = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        F = atom_in_fea.size(1)

        src = atom_in_fea.unsqueeze(1).expand(N, M, F)  # (N,M,F)
        nbr = atom_in_fea[nbr_fea_idx, :]               # (N,M,F)
        x = torch.cat([src, nbr, nbr_fea], dim=-1)      # (N,M,2F+B)

        msg = self.mlp(x)                               # (N,M,F)
        # padding mask (최종 합산에서 제거)
        mask = (nbr_fea_idx == 0).unsqueeze(-1)   # (N,M,1)
        msg = msg.masked_fill(mask, 0.0)
        msg = msg.sum(dim=1)                            # (N,F)

        out = atom_in_fea + msg                         # residual
        if self.use_bn:
            out = self.bn(out)
        out = self.act(out)
        return out

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 xrd=False, text=False, graph_type="cgcnn"):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        conv_to_fc_input_dim = atom_fea_len
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        if xrd:
            self.xrd_model = XRDFeatureExtractor(input_dim=128, output_dim=64, hidden_dim=128)
            conv_to_fc_input_dim += 64
        if text:
            self.text_model = TextFeatureExtractor(input_dim=768, output_dim=64, hidden_dim=128)
            conv_to_fc_input_dim += 64
        if graph_type == "cgcnn":
          self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        elif graph_type == "mpnn":
          self.convs = nn.ModuleList([MPNNLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        else:
          raise ValueError("Unknown graph type: {}".format(graph_type))
        self.conv_to_fc = nn.Linear(conv_to_fc_input_dim, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, xrd_feature=None, text_feature=None):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        # XRD feature extraction
        if hasattr(self, 'xrd_model'):
            xrd_fea = self.xrd_model(xrd_feature)
            crys_fea = torch.cat((crys_fea, xrd_fea), dim=1)

        # Text feature extraction
        if hasattr(self, 'text_model'):
            text_fea = self.text_model(text_feature)
            crys_fea = torch.cat((crys_fea, text_fea), dim=1)

        #crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


logger = logging.getLogger(__name__)

DEFAULT_ELEMENTS = (*list(DEFAULT_ELEMENTS[:83]), "Po", "At", "Rn", "Fr", "Ra", *list(DEFAULT_ELEMENTS[83:]))

class CHGNetLayer(nn.Module):
    """Main CHGNet model."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] | None = None,
        dim_atom_embedding: int = 64,
        dim_bond_embedding: int = 64,
        dim_angle_embedding: int = 64,
        dim_state_embedding: int | None = None,
        dim_state_feats: int | None = None,
        non_linear_bond_embedding: bool = False,
        non_linear_angle_embedding: bool = False,
        cutoff: float = 6.0,
        threebody_cutoff: float = 3.0,
        cutoff_exponent: int = 5,
        max_n: int = 9,
        max_f: int = 4,
        learn_basis: bool = True,
        num_blocks: int = 4,
        shared_bond_weights: Literal["bond", "three_body_bond", "both"] | None = "both",
        layer_bond_weights: Literal["bond", "three_body_bond", "both"] | None = None,
        atom_conv_hidden_dims: Sequence[int] = (64,),
        bond_update_hidden_dims: Sequence[int] | None = None,
        bond_conv_hidden_dims: Sequence[int] = (64,),
        angle_update_hidden_dims: Sequence[int] | None = (),
        conv_dropout: float = 0.0,
        pooling_operation: Literal["sum", "mean"] = "sum",
        readout_field: Literal["atom_feat", "bond_feat", "angle_feat"] = "atom_feat",
        activation_type: str = "swish",
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        num_site_targets: int = 1,
        **kwargs,
    ):
        """
        Args:
            element_types (list(str)): List of element types to consider in the model.
                If None, defaults to DEFAULT_ELEMENTS.
                Default = None
            dim_atom_embedding (int): Dimension of the atom embedding.
                Default = 64
            dim_bond_embedding (int): Dimension of the bond embedding.
                Default = 64
            dim_angle_embedding (int): Dimension of the angle embedding.
                Default = 64
            dim_state_embedding (int): Dimension of the state embedding.
                Default = None
            dim_state_feats (int): Dimension of the state features.
                Default = None
            non_linear_bond_embedding (bool): Whether to use a non-linear embedding for the
                bond features (single layer MLP).
                Default = False
            non_linear_angle_embedding (bool): Whether to use a non-linear embedding for the
                angle features (single layer MLP).
                Default = False
            cutoff (float): cutoff radius of atom graph.
                Default = 6.0
            threebody_cutoff (float): cutoff radius for bonds included in bond graph for
                three body interactions.
                Default = 3.0
            cutoff_exponent (int): exponent for the smoothing cutoff function.
                Default = 5
            max_n (int): maximum number of basis functions for the bond length radial expansion.
                Default = 9
            max_f (int): maximum number of basis functions for the angle Fourier expansion.
                Default = 4
            learn_basis (bool): whether to learn the frequencies used in basis functions
                or use fixed basis functions.
                Default = True
            num_blocks (int): number of graph convolution blocks.
                Default = 4
            shared_bond_weights (str): whether to share bond weights among layers in the atom
                and bond graph blocks.
                Default = "both"
            layer_bond_weights (str): whether to use independent weights in each convolution layer.
                Default = None
            atom_conv_hidden_dims (Sequence(int)): hidden dimensions for the atom graph convolution layers.
                Default = (64, )
            bond_update_hidden_dims: (Sequence(int)) hidden dimensions for the atom to
                bond message passing layer in atom graph. None means no atom to bond
                message passing layer in atom graph.
                Default = None
            bond_conv_hidden_dims (Sequence(int)): hidden dimensions for the bond graph convolution layers.
                Default = (64, )
            angle_update_hidden_dims (Sequence(int)): hidden dimensions for the angle layer.
                Default = ()
            conv_dropout (float): dropout probability for the graph convolution layers.
                Default = 0.0
            pooling_operation (str): type of readout pooling operation to use.
                Default = "sum"
            readout_field (str): field to readout from the graph.
                Default = "atom_feat"
            activation_type (str): activation function to use.
                Default = "swish"
            normalization (str): Normalization type to use in update functions.
                Either "graph" or "layer". If None, no normalization is applied.
                Default = None
            normalize_hidden (bool): whether to normalize the hidden layers in
                convolution update functions.
                Default = False
            num_site_targets (int): number of site-wise targets to predict. (ie magnetic moments)
                Default = 1
            **kwargs: additional keyword arguments.
        """
        super().__init__()

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        element_types = DEFAULT_ELEMENTS
        self.use_bond_graph = threebody_cutoff > 0
        if not self.use_bond_graph and readout_field == "angle_feat":
            raise ValueError(
                f"Angle Readout requires threebody_cutoff > 0, "
                f"but CHGNet is initialized with threebody_cutoff={threebody_cutoff}"
            )

        # basis expansions for bond lengths, triple interaction bond lengths and angles
        self.bond_expansion = RadialBesselFunction(max_n=max_n, cutoff=cutoff, learnable=learn_basis)
        self.threebody_bond_expansion = (
            RadialBesselFunction(max_n=max_n, cutoff=threebody_cutoff, learnable=learn_basis)
            if self.use_bond_graph
            else None
        )
        self.angle_expansion = FourierExpansion(max_f=max_f, learnable=learn_basis) if self.use_bond_graph else None

        # embedding block for atom, bond, angle, and optional state features
        self.include_states = dim_state_feats is not None
        self.state_embedding = nn.Embedding(dim_state_feats, dim_state_embedding) if self.include_states else None  # type: ignore[arg-type]
        self.atom_embedding = nn.Embedding(len(element_types), dim_atom_embedding)
        self.bond_embedding = MLP_norm(
            [max_n, dim_bond_embedding], activation=activation, activate_last=non_linear_bond_embedding, bias_last=False
        )
        self.angle_embedding = (
            MLP_norm(
                [2 * max_f + 1, dim_angle_embedding],
                activation=activation,
                activate_last=non_linear_angle_embedding,
                bias_last=False,
            )
            if self.use_bond_graph
            else None
        )

        # shared message bond distance smoothing weights
        self.atom_bond_weights = (
            nn.Linear(max_n, dim_atom_embedding, bias=False) if shared_bond_weights in ["bond", "both"] else None
        )
        self.bond_bond_weights = (
            nn.Linear(max_n, dim_bond_embedding, bias=False) if shared_bond_weights in ["bond", "both"] else None
        )
        self.threebody_bond_weights = (
            nn.Linear(max_n, dim_bond_embedding, bias=False)
            if shared_bond_weights in ["three_body_bond", "both"] and self.use_bond_graph
            else None
        )

        # operations involving the graph (i.e. atom graph) to update atom and bond features
        self.atom_graph_layers = nn.ModuleList(
            [
                CHGNetAtomGraphBlock(
                    num_atom_feats=dim_atom_embedding,
                    num_bond_feats=dim_bond_embedding,
                    atom_hidden_dims=atom_conv_hidden_dims,
                    bond_hidden_dims=bond_update_hidden_dims,
                    num_state_feats=dim_state_embedding,
                    activation=activation,
                    normalization=normalization,
                    normalize_hidden=normalize_hidden,
                    dropout=conv_dropout,
                    rbf_order=max_n if layer_bond_weights in ["bond", "both"] else 0,
                )
                for _ in range(num_blocks)
            ]
        )

        # operations involving the line graph (i.e. bond graph) to update bond and angle features
        self.bond_graph_layers = (
            nn.ModuleList(
                [
                    CHGNetBondGraphBlock(
                        num_atom_feats=dim_atom_embedding,
                        num_bond_feats=dim_bond_embedding,
                        num_angle_feats=dim_angle_embedding,
                        bond_hidden_dims=bond_conv_hidden_dims,
                        angle_hidden_dims=angle_update_hidden_dims,
                        activation=activation,
                        normalization=normalization,
                        normalize_hidden=normalize_hidden,
                        bond_dropout=conv_dropout,
                        angle_dropout=conv_dropout,
                        rbf_order=max_n if layer_bond_weights in ["three_body_bond", "both"] else 0,
                    )
                    for _ in range(num_blocks - 1)
                ]
            )
            if self.use_bond_graph
            else None
        )

        self.sitewise_readout = (
            nn.Linear(dim_atom_embedding, num_site_targets) if num_site_targets > 0 else lambda x: None
        )

        self.element_types = element_types
        self.max_n = max_n
        self.max_f = max_f
        self.cutoff = cutoff
        self.cutoff_exponent = cutoff_exponent
        self.three_body_cutoff = threebody_cutoff

        self.n_blocks = num_blocks
        self.readout_operation = pooling_operation
        self.readout_field = readout_field

    def forward(
        self,
        g: dgl.DGLGraph,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
        error_handling: bool = True,
    ):
        """Forward pass of the model.

        Args:
            g (dgl.DGLGraph): Input g.
            state_attr (torch.Tensor, optional): State features. Defaults to None.
            l_g (dgl.DGLGraph, optional): Line graph. Defaults to None and is computed internally.
            error_handling (bool, optional): Whether to allow numerical tolerance when an error occurs in
                l_g construction. Defaults to True.

        Returns:
            torch.Tensor: Model output.
        """
        # compute bond vectors and distances and add to g, needs to be computed here to register gradients
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec.to(g.device)
        g.edata["bond_dist"] = bond_dist.to(g.device)
        bond_expansion = self.bond_expansion(bond_dist)
        smooth_cutoff = polynomial_cutoff(bond_expansion, self.cutoff, self.cutoff_exponent)
        g.edata["bond_expansion"] = smooth_cutoff * bond_expansion

        # compute state, atom, bond and angle embeddings
        atom_features = self.atom_embedding(g.ndata["node_type"])
        bond_features = self.bond_embedding(g.edata["bond_expansion"])
        if self.state_embedding is not None and state_attr is not None:
            state_attr = self.state_embedding(state_attr)
        else:
            state_attr = None

        # create bond graph (line graph) with necessary node and edge data
        if self.use_bond_graph:
            if l_g is None:
                bond_graph = create_line_graph(g, self.three_body_cutoff, directed=True, error_handling=error_handling)
            else:
                # need to ensure the line graph matches the graph
                bond_graph = ensure_line_graph_compatibility(g, l_g, self.three_body_cutoff, directed=True)

            bond_graph.ndata["bond_index"] = bond_graph.ndata["edge_ids"]
            threebody_bond_expansion = self.threebody_bond_expansion(bond_graph.ndata["bond_dist"])  # type: ignore[misc]
            smooth_cutoff = polynomial_cutoff(threebody_bond_expansion, self.three_body_cutoff, self.cutoff_exponent)
            bond_graph.ndata["bond_expansion"] = smooth_cutoff * threebody_bond_expansion
            # the center atom is the dst atom of the src bond or the reverse (the src atom of the dst bond)
            # need to use "bond_index" just to be safe always
            bond_indices = bond_graph.ndata["bond_index"][bond_graph.edges()[0]]
            bond_graph.edata["center_atom_index"] = g.edges()[1][bond_indices]
            bond_graph.apply_edges(compute_theta)
            bond_graph.edata["angle_expansion"] = self.angle_expansion(bond_graph.edata["theta"])  # type: ignore[misc]
            angle_features = self.angle_embedding(bond_graph.edata["angle_expansion"])  # type: ignore[misc]
        else:
            bond_graph = None
            angle_features = None

        # shared message weights
        atom_bond_weights = (
            self.atom_bond_weights(g.edata["bond_expansion"]) if self.atom_bond_weights is not None else None
        )
        bond_bond_weights = (
            self.bond_bond_weights(g.edata["bond_expansion"]) if self.bond_bond_weights is not None else None
        )
        threebody_bond_weights = (
            self.threebody_bond_weights(bond_graph.ndata["bond_expansion"])
            if self.threebody_bond_weights is not None
            else None
        )

        # message passing layers
        for i in range(self.n_blocks - 1):
            atom_features, bond_features, state_attr = self.atom_graph_layers[i](
                g, atom_features, bond_features, state_attr, atom_bond_weights, bond_bond_weights
            )
            if self.use_bond_graph:
                bond_features, angle_features = self.bond_graph_layers[i](  # type: ignore
                    bond_graph, atom_features, bond_features, angle_features, threebody_bond_weights
                )

        # site wise target readout
        g.ndata["magmom"] = self.sitewise_readout(atom_features)

        # last atom graph message passing layer
        atom_features, bond_features, state_attr = self.atom_graph_layers[-1](
            g, atom_features, bond_features, state_attr, atom_bond_weights, bond_bond_weights
        )

        if self.readout_field == "atom_feat":
            g.ndata["atom_hidden"] = atom_features  # (N_total_atoms, dim_atom_embedding)
            crys_fea = readout_nodes(g, "atom_hidden", op=self.readout_operation)  # (N0, dim_atom_embedding)

        elif self.readout_field == "bond_feat":
            g.edata["bond_hidden"] = bond_features
            crys_fea = readout_edges(g, "bond_hidden", op=self.readout_operation)  # (N0, dim_bond_embedding)

        else:  # self.readout_field == "angle_feat"
            bond_graph.edata["angle_hidden"] = angle_features
            crys_fea = readout_edges(bond_graph, "angle_hidden", op=self.readout_operation)

        return crys_fea  # (N0, hidden_dim)

class M3GNetLayer(nn.Module):
    def __init__(
        self,
        element_types,
        atom_fea_len=64,
        cutoff=5.0,
        threebody_cutoff=4.0,
        **m3gnet_kwargs,
    ):
        super().__init__()
        self.m3gnet = M3GNet(
            element_types=tuple(element_types),
            dim_node_embedding=atom_fea_len,
            dim_edge_embedding=atom_fea_len,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            is_intensive=True,             
            readout_type="weighted_atom",
            task_type="regression",  
            ntargets=1,                     
            **m3gnet_kwargs,
        )

    def forward(self, g, state_attr=None):
        fea_dict = self.m3gnet(
            g=g,
            state_attr=state_attr,
            return_all_layer_output=True,
        )
        crys_fea = fea_dict["readout"]   # (N0, readout_dim)
        return crys_fea


class MatglGraphConvNet(nn.Module):
    def __init__(self, element_types, atom_fea_len=64, h_fea_len=128, n_h=1, xrd=False, text=False, 
                 cutoff=6.0, graph_type="chgnet", threebody_cutoff=4.0,**backbone_kwargs,):
        
        super().__init__()
        self.graph_type = graph_type

        if graph_type == "chgnet":
            self.backbone = CHGNetLayer(
                element_types=tuple(element_types),
                dim_atom_embedding=atom_fea_len,
                cutoff=cutoff,
                **backbone_kwargs,
            )

        if graph_type == "m3gnet":
           self.backbone = M3GNetLayer(
                element_types=tuple(element_types),
                atom_fea_len=atom_fea_len,
                cutoff=cutoff,
                threebody_cutoff=threebody_cutoff,
                **backbone_kwargs,
            )

        conv_to_fc_input_dim = atom_fea_len

        if xrd:
            self.xrd_model = XRDFeatureExtractor(input_dim=128, output_dim=64, hidden_dim=128)
            conv_to_fc_input_dim += 64
        if text:
            self.text_model = TextFeatureExtractor(input_dim=768, output_dim=64, hidden_dim=128)
            conv_to_fc_input_dim += 64
        self.conv_to_fc = nn.Linear(conv_to_fc_input_dim, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, graph_state, xrd_feature=None, text_feature=None):
        g, state_feats = graph_state
        crys_fea = self.backbone(g, state_attr=state_feats)  # (N0, atom_fea_len)
    
        # XRD feature extraction
        if hasattr(self, 'xrd_model'):
            xrd_fea = self.xrd_model(xrd_feature)
            crys_fea = torch.cat((crys_fea, xrd_fea), dim=1)

        # Text feature extraction
        if hasattr(self, 'text_model'):
            text_fea = self.text_model(text_feature)
            crys_fea = torch.cat((crys_fea, text_fea), dim=1)

        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        return out
    
