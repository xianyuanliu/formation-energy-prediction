"""Matbench e_form benchmark runner.

Runs the 5-fold cross-validation required by Matbench using the models
developed in this project (CGCNN, MPNN, CHGNet, M3GNet, ALIGNN) with
optional space-group text embeddings.

Usage:
    python run_matbench.py --graph_type cgcnn --text True --epochs 100
    python run_matbench.py --graph_type chgnet --text True --epochs 50 --lr 0.001 --optim Adam
"""

import sys, types

# ---- dgl.graphbolt shim ----
sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))

# ---- torchdata.datapipes compatibility shim ----
from torch.utils.data.datapipes.datapipe import IterDataPipe

pkg = types.ModuleType("torchdata.datapipes")
pkg.__path__ = []
sys.modules.setdefault("torchdata.datapipes", pkg)
mod = types.ModuleType("torchdata.datapipes.iter")
mod.IterDataPipe = IterDataPipe
sys.modules["torchdata.datapipes.iter"] = mod

import argparse
import os
import time
import warnings
import json
import datetime
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matbench.bench import MatbenchBenchmark

from data import (
    GaussianDistance,
    AtomCustomJSONInitializer,
    collate_pool,
    collate_pool_matgl,
    collate_pool_alignn,
)
from models.cgcnn import CrystalGraphConvNet, MatglGraphConvNet, AlignnGraphConvNet
from models.sg_text_module import TextEmbeddingDataset
from utils.utils import Normalizer, mae, AverageMeter

# Load SpaceGroupDescriber/SpaceGroupEmbedder from a dot-named directory
import importlib.util as _ilu

_sg_emb_path = os.path.join(
    os.path.dirname(__file__),
    "data_preprocessing", "embedding_space_group.v2", "space_group_embedding.py",
)
_spec = _ilu.spec_from_file_location("space_group_embedding", _sg_emb_path)
_sg_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sg_mod)
SpaceGroupDescriber = _sg_mod.SpaceGroupDescriber
SpaceGroupEmbedder = _sg_mod.SpaceGroupEmbedder

warnings.filterwarnings("ignore", message=".*fractional coordinates rounded.*")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_all_elements(structures):
    """Collect all unique element symbols from a list of structures."""
    elements = set()
    for s in structures:
        for site in s:
            elements.add(site.specie.symbol)
    return tuple(sorted(elements))


def get_space_groups(structures):
    """Extract space group symbols for each structure."""
    sgs = []
    for s in tqdm(structures, desc="Extracting space groups", leave=False):
        try:
            sga = SpacegroupAnalyzer(s)
            sg = sga.get_space_group_symbol()
        except Exception:
            sg = "P1"
        sgs.append(sg)
    return sgs


def ensure_text_embeddings(text_data, space_groups, matbert_path=None):
    """Generate embeddings for any space groups missing from the loaded CSV.

    Uses SpaceGroupDescriber to produce natural-language descriptions and
    SpaceGroupEmbedder (MatBERT) to convert them to 768-dim vectors.
    The generated embeddings are added to ``text_data`` in-place and also
    appended to the CSV file so subsequent runs skip the generation step.

    Parameters
    ----------
    text_data : TextEmbeddingDataset
        Pre-loaded embedding dataset.
    space_groups : list[str]
        All space group symbols found in the Matbench data.
    matbert_path : str or None
        Path to MatBERT model directory.  When *None*, the default relative
        path ``data_preprocessing/embedding_space_group.v2/matbert-base-uncased``
        is used.
    """
    import pandas as pd

    existing_keys = set(text_data._keys)
    needed = sorted(set(space_groups) - existing_keys)
    if not needed:
        return

    print(f"  {len(needed)} new space groups need embeddings — generating with MatBERT...")

    if matbert_path is None:
        matbert_path = os.path.join(
            os.path.dirname(__file__),
            "data_preprocessing", "embedding_space_group.v2", "matbert-base-uncased",
        )

    describer = SpaceGroupDescriber()
    embedder = SpaceGroupEmbedder(model_path=matbert_path, batch_size=64)

    descs = [describer.generate_description(sg) for sg in needed]
    embs = embedder.get_embeddings(descs)  # list of np.ndarray (768,)

    # Add to the in-memory dataset
    for sg, emb in zip(needed, embs):
        text_data.text_embeddings[sg] = torch.tensor(emb, dtype=torch.float32)
    text_data._keys = list(text_data.text_embeddings.keys())

    # Append to the CSV so we don't regenerate next time
    csv_path = os.path.join("data", "space_group_embeddings.csv")
    if os.path.exists(csv_path):
        new_rows = pd.DataFrame(
            embs,
            index=needed,
            columns=[f"dim_{i}" for i in range(len(embs[0]))],
        )
        new_rows.index.name = "space_group"
        new_rows.to_csv(csv_path, mode="a", header=False)
        print(f"  Appended {len(needed)} new embeddings to {csv_path}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MatbenchDataset(Dataset):
    """Wraps Matbench Structure objects into a PyTorch Dataset compatible
    with the existing collate functions and model forward signatures.
    """

    def __init__(
        self,
        structures,
        targets,
        space_groups,
        base_data_dir="data",
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        graph_type="cgcnn",
        cutoff=6.0,
        element_types=None,
        use_text=True,
        text_data=None,
    ):
        self.structures = structures
        self.targets = targets
        self.space_groups = space_groups
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.graph_type = graph_type
        self.cutoff = cutoff
        self.use_text = use_text
        self.text_data = text_data

        # Atom feature initializer
        atom_init_file = os.path.join(base_data_dir, "atom_init.json")
        self.ari = AtomCustomJSONInitializer(atom_init_file)

        # Gaussian distance expansion (for cgcnn/mpnn)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

        # Element types
        if element_types is not None:
            self.element_types = tuple(element_types)
        else:
            self.element_types = get_all_elements(structures)

        # MatGL graph converter
        if self.graph_type in ("chgnet", "m3gnet"):
            from matgl.ext.pymatgen import Structure2Graph

            self.graph_converter = Structure2Graph(
                element_types=self.element_types,
                cutoff=self.cutoff,
            )

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        crystal = self.structures[idx]
        target = torch.Tensor([self.targets[idx]])
        space_group = self.space_groups[idx]
        cif_id = str(idx)

        # XRD disabled
        xrd_fea = torch.zeros(128)

        # Text embedding
        if self.use_text and self.text_data is not None:
            try:
                text_fea = self.text_data[space_group]
            except (KeyError, TypeError):
                text_fea = torch.zeros(768)
        else:
            text_fea = torch.zeros(768)

        # ---------- graph construction ----------
        if self.graph_type in ("cgcnn", "mpnn"):
            atom_fea = np.vstack(
                [self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]
            )
            atom_fea = torch.Tensor(atom_fea)
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    nbr_fea_idx.append(
                        list(map(lambda x: x[2], nbr))
                        + [-1] * (self.max_num_nbr - len(nbr))
                    )
                    nbr_fea.append(
                        list(map(lambda x: x[1], nbr))
                        + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                    )
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
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
            from jarvis.core.atoms import pmg_to_atoms
            from alignn.graphs import Graph

            jarvis_atoms = pmg_to_atoms(crystal)
            g, lg = Graph.atom_dgl_multigraph(
                jarvis_atoms,
                cutoff=self.radius,
                max_neighbors=self.max_num_nbr,
                compute_line_graph=True,
            )
            lattice_tensor = torch.tensor(jarvis_atoms.lattice_mat).float()
            return (g, lg, lattice_tensor), target, cif_id, space_group, xrd_fea, text_fea


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args, dataset):
    """Instantiate the appropriate model architecture."""
    if args.graph_type in ("cgcnn", "mpnn"):
        structures, _, _, _, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=args.atom_fea_len,
            n_conv=args.n_conv,
            h_fea_len=args.h_fea_len,
            n_h=args.n_h,
            xrd=False,
            text=args.text,
            graph_type=args.graph_type,
        )
    elif args.graph_type in ("chgnet", "m3gnet"):
        model = MatglGraphConvNet(
            element_types=dataset.element_types,
            atom_fea_len=args.atom_fea_len,
            h_fea_len=args.h_fea_len,
            n_h=args.n_h,
            xrd=False,
            text=args.text,
            cutoff=dataset.cutoff,
            threebody_cutoff=4.0,
            graph_type=args.graph_type,
        )
    elif args.graph_type == "alignn":
        model = AlignnGraphConvNet(
            atom_fea_len=92,
            edge_fea_len=80,
            triplet_fea_len=40,
            h_fea_len=args.h_fea_len,
            n_h=args.n_h,
            xrd=False,
            text=args.text,
        )
    else:
        raise ValueError(f"Unknown graph_type: {args.graph_type}")
    return model


# ---------------------------------------------------------------------------
# Training / inference helpers
# ---------------------------------------------------------------------------

def train_one_epoch(args, loader, model, criterion, optimizer, normalizer, device):
    model.train()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    for input_data, target, _, xrd_fea, text_fea in loader:
        if args.graph_type in ("cgcnn", "mpnn"):
            input_var = (
                input_data[0].to(device),
                input_data[1].to(device),
                input_data[2].to(device),
                [idx.to(device) for idx in input_data[3]],
                xrd_fea.to(device),
                text_fea.to(device),
            )
            target_normed = normalizer.norm(target).to(device)
            out, _ = model(*input_var)

        elif args.graph_type in ("chgnet", "m3gnet"):
            g, state = input_data
            g = g.to(device)
            state = state.to(device)
            target_normed = normalizer.norm(target).to(device)
            out, _ = model((g, state), xrd_fea.to(device), text_fea.to(device))

        elif args.graph_type == "alignn":
            bg, blg, blat = input_data
            bg = bg.to(device)
            blg = blg.to(device)
            blat = blat.to(device)
            target_normed = normalizer.norm(target).to(device)
            out, _ = model(bg, blg, blat, xrd_fea.to(device), text_fea.to(device))

        loss = criterion(out, target_normed)
        mae_err = mae(normalizer.denorm(out.data.cpu()), target)

        losses.update(loss.item(), target.size(0))
        mae_errors.update(mae_err.item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, mae_errors.avg


@torch.no_grad()
def evaluate(args, loader, model, normalizer, device):
    """Return (mae, list_of_predictions)."""
    model.eval()
    mae_errors = AverageMeter()
    all_preds = []

    for input_data, target, _, xrd_fea, text_fea in loader:
        if args.graph_type in ("cgcnn", "mpnn"):
            input_var = (
                input_data[0].to(device),
                input_data[1].to(device),
                input_data[2].to(device),
                [idx.to(device) for idx in input_data[3]],
                xrd_fea.to(device),
                text_fea.to(device),
            )
            out, _ = model(*input_var)

        elif args.graph_type in ("chgnet", "m3gnet"):
            g, state = input_data
            g = g.to(device)
            state = state.to(device)
            out, _ = model((g, state), xrd_fea.to(device), text_fea.to(device))

        elif args.graph_type == "alignn":
            bg, blg, blat = input_data
            bg = bg.to(device)
            blg = blg.to(device)
            blat = blat.to(device)
            out, _ = model(bg, blg, blat, xrd_fea.to(device), text_fea.to(device))

        pred = normalizer.denorm(out.data.cpu())
        mae_err = mae(pred, target)
        mae_errors.update(mae_err.item(), target.size(0))
        all_preds.append(pred.view(-1))

    all_preds = torch.cat(all_preds)
    return mae_errors.avg, all_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Matbench e_form benchmark")
    parser.add_argument("--base_data_dir", default="data")
    parser.add_argument("--graph_type", default="cgcnn", choices=["cgcnn", "mpnn", "chgnet", "m3gnet", "alignn"])
    parser.add_argument("--text", default=True, type=lambda v: v.lower() in ("true", "1", "yes"))
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lr-milestones", default=[80], nargs="+", type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--optim", default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--atom-fea-len", default=64, type=int)
    parser.add_argument("--h-fea-len", default=128, type=int)
    parser.add_argument("--n-conv", default=3, type=int)
    parser.add_argument("--n-h", default=1, type=int)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--result-dir", default="matbench_results")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument(
        "--matbert-path",
        default=os.path.join("data_preprocessing", "embedding_space_group.v2", "matbert-base-uncased"),
        help="Path to MatBERT model for generating missing space group embeddings",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    # Collate function
    collate_fns = {
        "cgcnn": collate_pool,
        "mpnn": collate_pool,
        "chgnet": collate_pool_matgl,
        "m3gnet": collate_pool_matgl,
        "alignn": collate_pool_alignn,
    }
    collate_fn = collate_fns[args.graph_type]

    # Load text embeddings
    text_data = None
    if args.text:
        text_file = os.path.join(args.base_data_dir, "space_group_embeddings.csv")
        if os.path.exists(text_file):
            text_data = TextEmbeddingDataset(csv_path=text_file)
            print(f"Loaded {len(text_data)} space group text embeddings")
        else:
            print(f"WARNING: {text_file} not found. Text features will be zeros.")

    # Matbench benchmark
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_e_form"])

    for task in mb.tasks:
        task.load()
        print(f"\n{'='*60}")
        print(f"Task: {task.dataset_name}  ({len(task.df)} samples)")
        print(f"{'='*60}")

        # Pre-compute all space groups once (keyed by DataFrame index)
        all_structures = task.df["structure"].tolist()
        print("Pre-computing space groups for the full dataset...")
        sg_list = get_space_groups(all_structures)
        sg_by_index = dict(zip(task.df.index, sg_list))

        # Ensure text embeddings exist for all space groups in Matbench data
        if args.text and text_data is not None:
            ensure_text_embeddings(text_data, sg_list, matbert_path=args.matbert_path)

        # Pre-compute all element types from the full dataset
        print("Collecting element types...")
        all_element_types = get_all_elements(all_structures)
        print(f"  {len(all_element_types)} unique elements found")

        for fold in task.folds:
            fold_start = time.time()
            print(f"\n--- Fold {fold} ---")

            # Get data
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            test_inputs = task.get_test_data(fold, include_target=False)

            train_structures = train_inputs.tolist()
            train_targets = train_outputs.tolist()
            test_structures = test_inputs.tolist()

            # Map original DataFrame indices to pre-computed space groups
            train_sgs = [sg_by_index[i] for i in train_inputs.index]
            test_sgs = [sg_by_index[i] for i in test_inputs.index]

            print(f"  Train: {len(train_structures)}, Test: {len(test_structures)}")

            # Create datasets
            dataset_kwargs = dict(
                base_data_dir=args.base_data_dir,
                graph_type=args.graph_type,
                cutoff=6.0,
                element_types=all_element_types,
                use_text=args.text,
                text_data=text_data,
            )

            full_train_dataset = MatbenchDataset(
                train_structures, train_targets, train_sgs, **dataset_kwargs
            )
            # For test, targets are just placeholders (zeros)
            test_dataset = MatbenchDataset(
                test_structures, [0.0] * len(test_structures), test_sgs, **dataset_kwargs
            )

            # Train / val split
            n_train = len(full_train_dataset)
            g = torch.Generator().manual_seed(fold)
            indices = torch.randperm(n_train, generator=g).tolist()
            val_size = max(1, int(n_train * args.val_ratio))
            train_idx_split = indices[val_size:]
            val_idx_split = indices[:val_size]

            train_loader = DataLoader(
                full_train_dataset,
                batch_size=args.batch_size,
                sampler=SubsetRandomSampler(train_idx_split),
                num_workers=args.workers,
                collate_fn=collate_fn,
                pin_memory=use_cuda,
            )
            val_loader = DataLoader(
                full_train_dataset,
                batch_size=args.batch_size,
                sampler=SubsetRandomSampler(val_idx_split),
                num_workers=args.workers,
                collate_fn=collate_fn,
                pin_memory=use_cuda,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=collate_fn,
                pin_memory=use_cuda,
            )

            # Normalizer from training targets
            if n_train < 500:
                sample_list = [full_train_dataset[i] for i in range(n_train)]
            else:
                sample_list = [full_train_dataset[i] for i in sample(range(n_train), 500)]
            sample_targets = torch.stack([d[1] for d in sample_list], dim=0)
            normalizer = Normalizer(sample_targets)

            # Build model
            model = build_model(args, full_train_dataset)
            model = model.to(device)

            # Optimizer & scheduler
            criterion = nn.MSELoss()
            if args.optim == "SGD":
                optimizer = optim.SGD(
                    model.parameters(), args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay,
                )
            else:
                optimizer = optim.Adam(
                    model.parameters(), args.lr, weight_decay=args.weight_decay,
                )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.lr_milestones, gamma=0.1,
            )

            # Training loop
            best_val_mae = 1e10
            best_state = None

            for epoch in range(args.epochs):
                train_loss, train_mae = train_one_epoch(
                    args, train_loader, model, criterion, optimizer, normalizer, device,
                )
                val_mae, _ = evaluate(args, val_loader, model, normalizer, device)
                scheduler.step()

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if (epoch + 1) % args.print_freq == 0 or epoch == 0:
                    print(
                        f"  Epoch {epoch+1:3d}/{args.epochs}  "
                        f"train_mae={train_mae:.4f}  val_mae={val_mae:.4f}  "
                        f"best_val_mae={best_val_mae:.4f}"
                    )

            # Load best model and predict
            model.load_state_dict(best_state)
            model = model.to(device)
            test_mae, predictions = evaluate(args, test_loader, model, normalizer, device)

            # Record to matbench
            task.record(fold, predictions.numpy())

            fold_time = time.time() - fold_start
            print(
                f"  Fold {fold} done.  test_mae={test_mae:.4f}  "
                f"time={datetime.timedelta(seconds=int(fold_time))}"
            )

            # Save fold checkpoint
            fold_path = os.path.join(args.result_dir, f"fold_{fold}_best.pth.tar")
            torch.save(
                {
                    "fold": fold,
                    "state_dict": best_state,
                    "normalizer": normalizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "args": vars(args),
                },
                fold_path,
            )

    # Save matbench results
    results_path = os.path.join(args.result_dir, "results.json.gz")
    mb.to_file(results_path)
    print(f"\nResults saved to {results_path}")

    # Print scores
    for task in mb.tasks:
        print(f"\n{task.dataset_name} scores:")
        print(task.scores)


if __name__ == "__main__":
    main()
