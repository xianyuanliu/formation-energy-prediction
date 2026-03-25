from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import dgl
from models.xrd_module import XRDDataset
from models.sg_text_module import TextEmbeddingDataset
# Optimized Environment-based Imports
try:
    from jarvis.core.atoms import pmg_to_atoms
    from alignn.graphs import Graph
    HAS_ALIGNN_DATA = True
except ImportError:
    HAS_ALIGNN_DATA = False

try:
    import os
    # MUST be set before importing matgl to load DGL modules correctly
    os.environ["MATGL_BACKEND"] = "dgl"
    import matgl
    # In MatGL 2.0.6, Structure2Graph is the correct proxy in matgl.ext.pymatgen
    from matgl.ext.pymatgen import Structure2Graph
    from matgl.graph.converters import GraphConverter
    HAS_MATGL_DATA = True
except Exception as e:
    print(f"[!] Critical: MatGL initialization failed in data.py: {e}")
    # import traceback
    # traceback.print_exc()
    HAS_MATGL_DATA = False

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    batch_xrd_fea = []
    batch_text_fea = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id, space_group, xrd_fea, text_fea) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        
        idx = nbr_fea_idx.clone()  # (n_i, M)
        mask = idx >= 0           
        idx[mask] += base_idx     
        batch_nbr_fea_idx.append(idx)
        
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        batch_xrd_fea.append(xrd_fea)
        batch_text_fea.append(text_fea)
        
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids,\
        torch.stack(batch_xrd_fea, dim=0), \
        torch.stack(batch_text_fea, dim=0)

def collate_pool_matgl(dataset_list):
    """
    MatGL's collate:
    ((graph, state_feats), target, cif_id, space_group, xrd_fea, text_fea)
    """
    graphs = []
    state_list = []
    batch_target = []
    batch_cif_ids = []
    batch_xrd_fea = []
    batch_text_fea = []

    for (graph_state, target, cif_id, space_group, xrd_fea, text_fea) in dataset_list:
        graph, state_feats = graph_state
        graphs.append(graph)
        state_list.append(state_feats)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        batch_xrd_fea.append(xrd_fea)
        batch_text_fea.append(text_fea)

    batch_graph = dgl.batch(graphs)
    batch_state = torch.stack(state_list, dim=0)        # (N0, state_dim or 1)
    batch_target = torch.stack(batch_target, dim=0)     # (N0, 1)
    batch_xrd_fea = torch.stack(batch_xrd_fea, dim=0)   # (N0, xrd_dim)
    batch_text_fea = torch.stack(batch_text_fea, dim=0) # (N0, text_dim)

    return (batch_graph, batch_state), batch_target, batch_cif_ids, batch_xrd_fea, batch_text_fea

def collate_pool_alignn(dataset_list):
    """
    ALIGNN collate function.

    Parameters
    ----------
    dataset_list : list
        List of dataset items, where each item has the form
        ((g, lg, lat), target, cif_id, space_group, xrd_fea, text_fea).
        Here ``g`` and ``lg`` are DGLGraphs, ``lat`` is a lattice tensor,
        ``target`` is the supervision target, ``cif_id`` is an identifier
        for the structure, and ``xrd_fea`` / ``text_fea`` are feature tensors.

    Returns
    -------
    tuple
        A 5-tuple:

        - (batch_g, batch_lg, batch_lat): batched graphs and lattice tensor, where
          ``batch_g`` and ``batch_lg`` are obtained via :func:`dgl.batch` and
          ``batch_lat`` is a tensor formed by stacking ``lat`` along dimension 0.
        - batch_target: tensor of stacked targets.
        - batch_cif_ids: list of CIF identifiers.
        - batch_xrd_fea: tensor of stacked XRD feature vectors.
        - batch_text_fea: tensor of stacked text feature vectors.
    """
    g_list, lg_list, lat_list = [], [], []
    batch_target, batch_cif_ids = [], []
    batch_xrd_fea, batch_text_fea = [], []

    for ((g, lg, lat), target, cif_id, space_group, xrd_fea, text_fea) in dataset_list:
        g_list.append(g)
        lg_list.append(lg)
        lat_list.append(lat)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        batch_xrd_fea.append(xrd_fea)
        batch_text_fea.append(text_fea)

    return (dgl.batch(g_list), dgl.batch(lg_list), torch.stack(lat_list, dim=0)), \
           torch.stack(batch_target, dim=0), \
           batch_cif_ids, \
           torch.stack(batch_xrd_fea, dim=0), \
           torch.stack(batch_text_fea, dim=0)

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of cifs files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a cifs file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, csv_filename, base_data_dir='data', cif_path=None, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=123, graph_type="cgcnn", cutoff=6.0, element_types=None,use_xrd=True, use_text=True):
        self.root_dir = root_dir
        self.cif_path = cif_path if cif_path else root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.graph_type = graph_type

        # Support for multiple CSV files
        if isinstance(csv_filename, str):
            csv_filenames = [csv_filename]
        elif csv_filename is None:
            raise ValueError("csv_filename must be provided (e.g., 'id_prop.csv')")
        else:
            csv_filenames = csv_filename

        self.id_prop_data = []
        for fname in csv_filenames:
            id_prop_file = os.path.join(self.root_dir, fname)
            if not os.path.exists(id_prop_file):
                raise FileNotFoundError(f"{id_prop_file} does not exist!")
            with open(id_prop_file) as f:
                reader = csv.DictReader(f)
                self.id_prop_data.extend([row for row in reader])

        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        if self.graph_type in ("cgcnn", "mpnn"):
            atom_init_file = os.path.join(base_data_dir, 'atom_init.json')
            if not os.path.exists(atom_init_file):
                raise FileNotFoundError(f"Missing essential file: {atom_init_file}")
            self.ari = AtomCustomJSONInitializer(atom_init_file)
            
        xrd_data_file = os.path.join(base_data_dir, 'XRD_data.csv')
        text_data_file = os.path.join(base_data_dir, 'space_group_embeddings.csv')
        self.use_xrd = use_xrd
        if self.use_xrd:
            if os.path.exists(xrd_data_file):
                self.xrd_data = XRDDataset(csv_path=xrd_data_file)
            else:
                warnings.warn(f"XRD data not found at {xrd_data_file}. Disabling XRD features.")
                self.use_xrd = False
                self.xrd_data = None
        else:
            self.xrd_data = None

        self.use_text = use_text
        if self.use_text:
            if os.path.exists(text_data_file):
                self.text_data = TextEmbeddingDataset(csv_path=text_data_file)
            else:
                warnings.warn(f"Text data not found at {text_data_file}. Disabling text features.")
                self.use_text = False
                self.text_data = None
        else:
            self.text_data = None

        if self.graph_type in ("cgcnn", "mpnn"):
            self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        # Strict column check
        if self.id_prop_data:
            required_cols = ['file_name', 'value_per_atom', 'space_group']
            first_row_keys = self.id_prop_data[0].keys()
            for col in required_cols:
                if col not in first_row_keys:
                    raise KeyError(f"Required column '{col}' is missing from the CSV file. (Current columns: {list(first_row_keys)})")

        element_set = set()
        for row in self.id_prop_data:
            cif_id = row['file_name']
            s = Structure.from_file(os.path.join(self.cif_path, cif_id + ".cif"))
            for site in s:
                element_set.add(site.specie.symbol)
        
        if element_types is None:
            element_set = set()
            for row in self.id_prop_data:
                cif_id = row['file_name']
                s = Structure.from_file(os.path.join(self.cif_path, cif_id + ".cif"))
                for site in s:
                    element_set.add(site.specie.symbol)
            self.element_types = tuple(sorted(element_set))
        else:
            self.element_types = tuple(element_types)
        
        self.cutoff = cutoff

        if self.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
            if not HAS_MATGL_DATA:
                raise ImportError("MatGL is required for chgnet/m3gnet/tensornet/qet but not installed.")
            # MatGL 2.0.6+: backend argument removed
            self.graph_converter = Structure2Graph(
                element_types=self.element_types,
                cutoff=self.cutoff,
            )

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        row  = self.id_prop_data[idx] 
        
        # Strict column access with English error messages
        try:
            cif_id = row['file_name']
            space_group = row['space_group']
            target_val = row['value_per_atom']
        except KeyError as e:
            raise KeyError(f"Required column missing while reading data item: {e}")

        target = torch.Tensor([float(target_val)])
        if self.use_xrd:
            xrd_fea = self.xrd_data[cif_id]
        else:
            xrd_fea = torch.zeros(128) 
        if self.use_text:
            text_fea = self.text_data[space_group]
        else:
            text_fea = torch.zeros(768)  

        crystal = Structure.from_file(os.path.join(self.cif_path, cif_id+'.cif'))
        if self.graph_type in ("cgcnn", "mpnn"):
            atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
            atom_fea = torch.Tensor(atom_fea)
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    warnings.warn('{} not find enough neighbors to build graph. '
                                'If it happens frequently, consider increase '
                                'radius.'.format(cif_id))
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                    [-1] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                [self.radius + 1.] * (self.max_num_nbr -
                                                        len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
            return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id, space_group, xrd_fea, text_fea
        
        elif self.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
            # MatGL DGL Structure2Graph returns (graph, lattice, state_feats)
            # lattice is returned as (1, 3, 3), we need (3, 3) for standard ops
            graph, lattice_mat, state_feats_default = self.graph_converter.get_graph(crystal)
            graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"].float(), lattice_mat[0])
            graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice_mat[0]
            state_feats = torch.tensor(state_feats_default)
            return (graph, state_feats), target, cif_id, space_group, xrd_fea, text_fea

        elif self.graph_type in ("alignn"):
            if not HAS_ALIGNN_DATA:
                raise ImportError("ALIGNN/Jarvis-tools is required for alignn but not installed.")
            jarvis_atoms = pmg_to_atoms(crystal)
            g, lg = Graph.atom_dgl_multigraph(
                jarvis_atoms, 
                cutoff=self.radius, 
                max_neighbors=self.max_num_nbr,
                compute_line_graph=True
            )
            lattice = torch.tensor(jarvis_atoms.lattice_mat).float()
            return (g, lg, lattice), target, cif_id, space_group, xrd_fea, text_fea


class MatbenchData(Dataset):
    def __init__(self, json_path, base_data_dir='data', max_num_nbr=12,
                 radius=8, dmin=0, step=0.2, random_seed=123,
                 graph_type="cgcnn", cutoff=6.0, element_types=None,
                 use_text=True):
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.graph_type = graph_type
        self.cutoff = cutoff

        with open(json_path, 'r') as f:
            raw = json.load(f)

        self.columns = raw['columns']
        struct_col = self.columns.index('structure')
        target_col = self.columns.index('e_form')

        self.entries = [
            (row[struct_col], row[target_col])
            for row in raw['data']
        ]

        random.seed(random_seed)
        random.shuffle(self.entries)

        atom_init_file = os.path.join(base_data_dir, 'atom_init.json')
        if not os.path.exists(atom_init_file):
            raise FileNotFoundError(f"Missing essential file: {atom_init_file}")
        self.ari = AtomCustomJSONInitializer(atom_init_file)

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        self.use_text = use_text
        if self.use_text:
            text_data_file = os.path.join(base_data_dir, 'space_group_embeddings.csv')
            if os.path.exists(text_data_file):
                self.text_data = TextEmbeddingDataset(csv_path=text_data_file)
            else:
                warnings.warn(f"Text data not found at {text_data_file}. Disabling text features.")
                self.use_text = False
                self.text_data = None
        else:
            self.text_data = None

        if element_types is not None:
            self.element_types = tuple(element_types)
        else:
            from matgl.config import DEFAULT_ELEMENTS
            self.element_types = tuple(DEFAULT_ELEMENTS)

        self._sg_cache = {}

        if self.graph_type in ("chgnet", "m3gnet"):
            from matgl.ext.pymatgen import Structure2Graph
            self.graph_converter = Structure2Graph(
                element_types=self.element_types,
                cutoff=self.cutoff,
            )

    def __len__(self):
        return len(self.entries)

    def _get_space_group(self, idx, structure):
        if idx in self._sg_cache:
            return self._sg_cache[idx]
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            sg_symbol = sga.get_space_group_symbol()
        except Exception:
            sg_symbol = "P1"
        self._sg_cache[idx] = sg_symbol
        return sg_symbol

    def __getitem__(self, idx):
        struct_dict, e_form = self.entries[idx]
        crystal = Structure.from_dict(struct_dict)
        target = torch.Tensor([float(e_form)])
        cif_id = str(idx)

        space_group = self._get_space_group(idx, crystal)

        if self.use_text:
            try:
                text_fea = self.text_data[space_group]
            except (KeyError, TypeError):
                text_fea = torch.zeros(self.text_data.num_text_features)
        else:
            text_fea = torch.zeros(768)

        xrd_fea = torch.zeros(128)

        if self.graph_type in ("cgcnn", "mpnn"):
            atom_fea = np.vstack([
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ])
            atom_fea = torch.Tensor(atom_fea)
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    warnings.warn(f'{cif_id} not find enough neighbors to build graph.')
                    nbr_fea_idx.append(
                        list(map(lambda x: x[2], nbr)) +
                        [-1] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(
                        list(map(lambda x: x[1], nbr)) +
                        [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)
            nbr_fea = torch.Tensor(nbr_fea)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
            return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id, space_group, xrd_fea, text_fea

        elif self.graph_type in ("chgnet", "m3gnet"):
            graph, lattice, state_feats_default = self.graph_converter.get_graph(crystal)
            graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
            graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice[0]
            state_feats = torch.tensor(state_feats_default)
            return (graph, state_feats), target, cif_id, space_group, xrd_fea, text_fea

        elif self.graph_type == "alignn":
            jarvis_atoms = pmg_to_atoms(crystal)
            g, lg = Graph.atom_dgl_multigraph(
                jarvis_atoms,
                cutoff=self.radius,
                max_neighbors=self.max_num_nbr,
                compute_line_graph=True
            )
            lattice = torch.tensor(jarvis_atoms.lattice_mat).float()
            return (g, lg, lattice), target, cif_id, space_group, xrd_fea, text_fea
