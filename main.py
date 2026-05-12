import sys, types
sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))

# ---- torchdata.datapipes compatibility shim (must be before importing dgl) ----
import sys, types
from torch.utils.data.datapipes.datapipe import IterDataPipe  

pkg = types.ModuleType("torchdata.datapipes")
pkg.__path__ = []  # mark as package-like
sys.modules.setdefault("torchdata.datapipes", pkg)

mod = types.ModuleType("torchdata.datapipes.iter")
mod.IterDataPipe = IterDataPipe
sys.modules["torchdata.datapipes.iter"] = mod
# -----------------------------------------------------------------------------

import os
import sys
import time
import datetime
import warnings
import yaml
import shutil
import glob
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR

from examples.config import get_cfg_defaults
from data import CIFData
from data import collate_pool, get_train_val_test_loader, collate_pool_matgl, collate_pool_alignn
from models.cgcnn import CrystalGraphConvNet, MatglGraphConvNet, AlignnGraphConvNet
from utils.utils import Normalizer, mae, save_checkpoint, AverageMeter
from thop import profile
import wandb

warnings.filterwarnings("ignore", message=".*fractional coordinates rounded.*")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_parse():
    """Parsing arguments with YAML"""
    import argparse
    parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')

    # 1. Add --config flag
    parser.add_argument('--config', default='', type=str, help='path to yaml config file')
    
    # Optuna parameters (Distributed SQLite)
    parser.add_argument('--optuna_study', default='', type=str, help='Optuna study DB url, e.g., sqlite:///optuna_hpo.db')
    parser.add_argument('--optuna_study_name', default='optuna_study', type=str, help='Optuna study name')
    parser.add_argument('--optuna_n_trials', default=1, type=int, help='Number of trials for this worker to run')

    # 2. Original Basic parameters
    parser.add_argument('--base_data_dir', default='data', help='path to common data files (atom_init, XRD, space_group_embeddings)')
    parser.add_argument('--data_path', default='data/split_both_hhi', help='path to csv files')
    parser.add_argument('--cif_path', default='data/cifs', help='path to cif files')
    parser.add_argument('--task', default='regression')
    parser.add_argument('--xrd', default=True, type=str2bool, help='use xrd features')
    parser.add_argument('--text', default=True, type=str2bool, help='use text features')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run (default: 30)')
    parser.add_argument('--use_early_stopping', default=False, type=str2bool, help='use early stopping (default: False)')
    parser.add_argument('--patience', default=30, type=int, help='early stopping patience (default: 30)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int, metavar='N', help='milestones for scheduler (default: [100])')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_file', default=None, nargs='+', help='train csv file name(s) in data_path')
    parser.add_argument('--test_file', default=None, nargs='+', help='test csv file name(s) in data_path')
    
    # 3. Original WandB parameters
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB for logging')
    parser.add_argument('--wandb_project', default='formation-energy-krict', type=str, help='WandB project name')
    parser.add_argument('--wandb_group', default='baseline', type=str, help='WandB group name')
    parser.add_argument('--wandb_name', default=None, type=str, help='WandB run name (None = auto-generated)')
    parser.add_argument('--online_mode', default=False, type=str2bool, help='Use WandB online mode for web monitoring (default: False)')

    # 4. Original Data split
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('--train-ratio', default=None, type=float, metavar='N', help='number of training data to be loaded (default none)')
    train_group.add_argument('--train-size', default=None, type=int, metavar='N', help='number of training data to be loaded (default none)')

    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N', help='percentage of validation data to be loaded (default 0.1)')
    valid_group.add_argument('--val-size', default=None, type=int, metavar='N', help='number of validation data to be loaded (default 1000)')

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N', help='percentage of test data to be loaded (default 0.1)')
    test_group.add_argument('--test-size', default=None, type=int, metavar='N', help='number of test data to be loaded (default 1000)')

    parser.add_argument('--result_dir', default=None, type=str, help='Directory to save results. If None, no folder is created.')
    parser.add_argument('--result_files', default=['checkpoint.pth.tar', 'model_best.pth.tar', 'test_results.csv'], nargs='+', help='List of files to move to result_dir')
    
    # 5. Original Model parameters
    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD', help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N', help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=128, type=int, metavar='N', help='number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N', help='number of conv layers')
    parser.add_argument('--n-h', default=1, type=int, metavar='N', help='number of hidden layers after pooling')
    parser.add_argument('--best_mae_error', default=1e10, type=float, metavar='N', help='best mae error (default: 1e10)')
    parser.add_argument('--graph_type', default="cgcnn", type=str, metavar="GRAPH", help='type of graph convolutional network (mpnn, cgcnn, chgnet, m3gnet, alignn, tensornet, qet)')
    
    args = parser.parse_args()

    # 6. Load YAML and apply dynamic logic (keeps main.py clean)
    from examples.config import load_and_apply_config
    load_and_apply_config(args)

    return args

best_mae_error = 1e10

def rmse(pred, target):
    # pred/target: torch.Tensor (N,) on CPU
    pred = pred.view(-1).numpy()
    target = target.view(-1).numpy()
    return float(np.sqrt(np.mean((pred - target) ** 2)))

def r2_score(pred, target):
    pred = pred.view(-1).numpy()
    target = target.view(-1).numpy()
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def main(args=None):
    global best_mae_error
    if args is None:
        args = arg_parse()

    if args.result_dir:
        out_dir = args.result_dir
        os.makedirs(out_dir, exist_ok=True)
        print(f"=> Result directory: {out_dir}")
    else:
        out_dir = "."

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
            mode="online" if args.online_mode else "offline",
            settings=wandb.Settings(console="off")
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("learning_rate", step_metric="epoch")    
    
    start_total_time = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for training but no CUDA device was found. Strict GPU execution is enforced.")
    device = torch.device("cuda")

    if args.graph_type in ("cgcnn", "mpnn"):
        collate_fn = collate_pool
    elif args.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
        collate_fn = collate_pool_matgl
    elif args.graph_type == "alignn":
        collate_fn = collate_pool_alignn

    # Data loader generation (Conditional branching)
    if args.train_file or args.test_file:
        # Mode A: Use separate files for training and testing
        if args.test_file and not args.train_file:
            all_csvs = [f for f in os.listdir(args.data_path) if f.endswith('.csv')]
            train_files = [f for f in all_csvs if f not in args.test_file]
            aux_files = ['XRD_data.csv', 'space_group_embeddings.csv']
            train_files = [f for f in train_files if f not in aux_files]
            test_files = args.test_file
            print(f"=> Auto-detected train files: {train_files}")
        else:
            train_files = args.train_file if args.train_file else []
            test_files = args.test_file if args.test_file else []
        print(f"=> Separate file mode: {args.train_file} (train) / {args.test_file} (test)")

        data_kwargs = {
            "root_dir": args.data_path,
            "cif_path": args.cif_path,
            "base_data_dir": args.base_data_dir,
            "graph_type": args.graph_type,
            "use_xrd": args.xrd,
            "use_text": args.text
        }

        tmp_train = CIFData(csv_filename=train_files, **data_kwargs)
        tmp_test = CIFData(csv_filename=test_files, **data_kwargs)
        element_types_union = sorted(set(tmp_train.element_types) | set(tmp_test.element_types))
        print(f"[Element Types Union Size] {len(element_types_union)}")

        full_train_dataset = CIFData(csv_filename=train_files, element_types=element_types_union, **data_kwargs)
        test_dataset = CIFData(csv_filename=test_files, element_types=element_types_union, **data_kwargs)

        g = torch.Generator()
        g.manual_seed(0)
        indices = torch.randperm(len(full_train_dataset), generator=g).tolist()
        
        # Calculate indices for validation split from the train file
        val_size = max(1, int(len(full_train_dataset) * args.val_ratio))
        train_idx = indices[val_size:]
        val_idx   = indices[:val_size]
        
        # Set up samplers for random split
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler   = SubsetRandomSampler(val_idx)
        
        # Create DataLoaders
        train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=args.workers,
                                  collate_fn=collate_fn, pin_memory=True)
        
        val_loader = DataLoader(full_train_dataset, batch_size=args.batch_size,
                                sampler=val_sampler, num_workers=args.workers,
                                collate_fn=collate_fn, pin_memory=True)
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers,
                                 collate_fn=collate_fn, pin_memory=True)
        
        # Set representative dataset for model building
        dataset = full_train_dataset

    else:
        # Mode B: Single file mode with ratio split
        print("=> Ratio split mode: Splitting one file into train/val/test")
        if not args.train_file:
             raise ValueError("Please provide a CSV file (e.g., --train_file my_data.csv) to split by ratio.")
        
        target_csv = args.train_file[0]
        dataset = CIFData(args.data_path, csv_filename=target_csv, base_data_dir=args.base_data_dir, graph_type=args.graph_type, use_xrd=args.xrd, use_text=args.text)
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=True,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            return_test=True)

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
            
        sample_target = torch.stack([d[1] for d in sample_data_list], dim=0)
        normalizer = Normalizer(sample_target)

    # build model
    if args.graph_type in ("cgcnn", "mpnn"):
        structures, _, _, _, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    xrd=args.xrd,
                                    text=args.text,
                                    graph_type=args.graph_type)
        
    elif args.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
        model = MatglGraphConvNet(
            element_types=dataset.element_types,
            atom_fea_len=args.atom_fea_len,
            h_fea_len=args.h_fea_len,
            n_h=args.n_h,
            xrd=args.xrd,
            text=args.text,
            cutoff=dataset.cutoff,
            threebody_cutoff=4.0,            # M3GNet 3-body cutoff
            graph_type=args.graph_type,
        )

    elif args.graph_type in ("alignn"):
        model = AlignnGraphConvNet(
            atom_fea_len=92,    # JARVIS default value example
            edge_fea_len=80,    # JARVIS default value example
            triplet_fea_len=40,
            h_fea_len=args.h_fea_len,
            n_h=args.n_h,
            xrd=args.xrd,
            text=args.text
        )
    else:
        raise ValueError(f"Unknown graph_type: {args.graph_type}")
        
    model.to(device)

    print("\n" + "="*30)
    print("      Calculating FLOPs")
    print("="*30)

    model.eval() 
    try:
        sample_data = next(iter(train_loader))
        if args.graph_type in ("cgcnn", "mpnn"):
            inputs = (
                sample_data[0][0].to(device), # atom_fea
                sample_data[0][1].to(device), # nbr_fea
                sample_data[0][2].to(device), # nbr_fea_idx
                [idx.to(device) for idx in sample_data[0][3]], # crystal_atom_idx
                sample_data[3].to(device),    # xrd_feature
                sample_data[4].to(device)     # text_feature
            )
        elif args.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
            graph_state = (
                sample_data[0][0].to(device), # batch_graph
                sample_data[0][1].to(device)  # state_feats
            )
            inputs = (
                graph_state,
                sample_data[3].to(device),    # xrd_feature
                sample_data[4].to(device)     # text_feature
            )
        elif args.graph_type in ("alignn"):
            batch_g, batch_lg, batch_lat = sample_data[0] 
            inputs = (
                batch_g.to(device),
                batch_lg.to(device),
                batch_lat.to(device),  
                sample_data[3].to(device), # xrd_feature
                sample_data[4].to(device)  # text_feature
            )
        flops, params = profile(model, inputs=inputs, verbose=False)
        if args.use_wandb:
            wandb.run.summary.update({
                "total_flops_g": flops / 1e9,
                "total_params_m": params / 1e6
            })
        
        print(f"[*] Batch Size: {args.batch_size}")
        print(f"[*] Total FLOPs for one batch: {flops / 1e9:.4f} GFLOPs")
        print(f"[*] FLOPs per crystal: {(flops / args.batch_size) / 1e6:.2f} MFLOPs")
        print(f"[*] Total Params: {params / 1e6:.2f} M")
    except Exception as e:
        print(f"[!] FLOPs calculation failed: {e}")
    
    print("="*30 + "\n")
    model.train()

    # define loss func and optimizer
    criterion = nn.SmoothL1Loss(beta=1.0)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    patience_counter = 0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss_avg, train_mae_avg = train(args, train_loader, model, criterion, optimizer, epoch, normalizer, device)

        # evaluate on validation set
        val_loss_avg, val_mae_avg = validate(args, val_loader, model, criterion, normalizer, device)
        mae_error = val_mae_avg
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "val/mae": mae_error,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train/loss": train_loss_avg,
                "train/mae": train_mae_avg,
                "val/loss": val_loss_avg,
                "val/mae": val_mae_avg,
            },step=epoch)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = mae_error < best_mae_error
        
        if args.use_early_stopping and args.patience > 0:
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1

        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)
        
        if args.use_early_stopping and args.patience > 0:
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] Validation MAE no improvement for {args.patience} epochs. Stopping early at epoch {epoch}.")
                break

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(args, test_loader, model, criterion, normalizer, device, test=True)
    
    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    total_time_str = str(datetime.timedelta(seconds=int(total_duration)))
    if args.use_wandb:
        wandb.run.summary["total_training_time_sec"] = total_duration
        wandb.run.summary["best_mae_error"] = best_mae_error
        wandb.finish() 

    print("\n" + "="*30)
    print(f"  Training Completed!")
    print(f"  Total Time Elapsed: {total_time_str}")
    print(f"  ({total_duration:.2f} seconds)")
    print("="*30)
    
    print("\n" + "-"*30)
    print("      Starting Post-Training Analysis")
    print("-"*30)
    try:
        ytr, ypr, emb_tr, _ = collect_pred_emb(args, train_loader, model, normalizer, device)
        yte, ype, emb_te, test_ids = collect_pred_emb(args, test_loader, model, normalizer, device)

        np.save(os.path.join(out_dir, "train_emb.npy"), emb_tr)
        np.save(os.path.join(out_dir, "test_emb.npy"), emb_te)
        print(f"=> Saved embeddings to {out_dir}")

        fig_path = os.path.join(out_dir, "ood_umap_kde.png")
        stats = make_ood_umap_figure(
            train_emb=emb_tr,
            test_emb=emb_te,
            y_true_test=yte,
            y_pred_test=ype,
            out_png=fig_path,
            density_threshold=1e-3, 
            n_neighbors=50,          
            min_dist=0.1             
        )
        print(f"=> Generated UMAP figure at {fig_path}")
    except Exception as e:
        print(f"[!] Warning: Post-training analysis failed: {e}")
    
    
    def collect_outputs(target_dir, files_to_move):
        print("\n" + "="*30)
        print(f"  Collecting Outputs to: {target_dir}")
        print("="*30)
        
        if target_dir == "." or not target_dir:
            print("=> Output directory is current directory, skipping move.")
            return
            
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            print(f"=> Created output directory: {target_dir}")

        # Ensure files_to_move is a flat list of patterns
        if isinstance(files_to_move, str):
            files_to_move = files_to_move.split()
        
        total_moved = 0
        for pattern in files_to_move:
            print(f"[*] Processing pattern: {pattern}")
            found_files = glob.glob(pattern)
            if not found_files:
                print(f"    - No files found matching '{pattern}' in {os.getcwd()}")
                continue
                
            for f in found_files:
                if os.path.exists(f):
                    dest = os.path.join(target_dir, os.path.basename(f))
                    if os.path.abspath(f) != os.path.abspath(dest):
                        try:
                            print(f"    - Moving: {f} -> {dest}")
                            shutil.move(f, dest)
                            total_moved += 1
                        except Exception as e:
                            print(f"    [!] Error moving {f}: {e}")
                    else:
                        print(f"    - File {f} is already in target directory.")
        print(f"\n=> Finished collecting outputs. Total files moved: {total_moved}")
        print("="*30 + "\n")

    collect_outputs(out_dir, args.result_files)
    
    # Return the objective value for Optuna optimization
    return best_mae_error


def train(args, train_loader, model, criterion, optimizer, epoch, normalizer, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    mre_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _, xrd_fea, text_fea) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.graph_type in ("cgcnn", "mpnn"):
            input_var = (input[0].to(device, non_blocking=True),
                        input[1].to(device, non_blocking=True),
                        input[2].to(device, non_blocking=True),
                        [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]],
                        xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None,
                        text_fea.to(device, non_blocking=True) if text_fea is not None else None)
        elif args.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
            batch_graph, batch_state = input
            batch_graph = batch_graph.to(device) 
            batch_state = batch_state.to(device, non_blocking=True)
            xrd_fea = xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None
            text_fea = text_fea.to(device, non_blocking=True) if text_fea is not None else None
            input_var = ((batch_graph, batch_state), xrd_fea, text_fea)
        elif args.graph_type == "alignn":
            batch_g, batch_lg, batch_lat = input
            batch_g = batch_g.to(device)
            batch_lg = batch_lg.to(device)
            batch_lat = batch_lat.to(device)
            xrd_fea = xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None
            text_fea = text_fea.to(device, non_blocking=True) if text_fea is not None else None
            input_var = (batch_g, batch_lg, batch_lat, xrd_fea, text_fea)
        else:
            raise ValueError(f"Unknown graph_type: {args.graph_type}")
        
        target_normed = normalizer.norm(target)
        target_var = target_normed.to(device, non_blocking=True)

        if args.graph_type in ("cgcnn", "mpnn"):
            out, emb = model(*input_var)
        elif args.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
            graph_state, xrd_in, text_in = input_var
            out, emb = model(graph_state, xrd_in, text_in)
        elif args.graph_type in ("alignn"):
            batch_g, batch_lg, batch_lat, xrd_in, text_in = input_var
            out, emb = model(batch_g, batch_lg, batch_lat, xrd_in, text_in)

        loss = criterion(out, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(out.detach().cpu()), target)
        mre_error = mae_error / target.abs().mean()

        losses.update(loss.detach().cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        mre_errors.update(mre_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                  'MRE {mre_errors.val:.3f} ({mre_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors, mre_errors=mre_errors)
            )
            
    return float(losses.avg), float(mae_errors.avg)

def validate(args, val_loader, model, criterion, normalizer, device, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    mre_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids, xrd_fea, text_fea) in enumerate(val_loader):
        if args.graph_type in ("cgcnn", "mpnn"):
            with torch.no_grad():
                input_var = (input[0].to(device, non_blocking=True),
                            input[1].to(device, non_blocking=True),
                            input[2].to(device, non_blocking=True),
                            [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]],
                            xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None,
                            text_fea.to(device, non_blocking=True) if text_fea is not None else None)
        elif args.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
            graph_state = input 
            with torch.no_grad():
                g, state_feats = graph_state
                g = g.to(device)
                state_feats = state_feats.to(device, non_blocking=True)
                xrd_fea_cuda = xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None
                text_fea_cuda = text_fea.to(device, non_blocking=True) if text_fea is not None else None
                input_var = ((g, state_feats), xrd_fea_cuda, text_fea_cuda)
        elif args.graph_type in ("alignn"):
            batch_g, batch_lg, batch_lat = input
            with torch.no_grad():
                batch_g = batch_g.to(device)
                batch_lg = batch_lg.to(device)
                batch_lat = batch_lat.to(device)
                xrd_fea_cuda = xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None
                text_fea_cuda = text_fea.to(device, non_blocking=True) if text_fea is not None else None
                input_var = (batch_g, batch_lg, batch_lat, xrd_fea_cuda, text_fea_cuda)
        else:
            raise ValueError(f"Unknown graph_type: {args.graph_type}")
        
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        with torch.no_grad():
            target_var = target_normed.to(device, non_blocking=True)

        # compute output
        out, emb = model(*input_var)
        loss = criterion(out, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(out.detach().cpu()), target)
        mre_error = mae_error / target.abs().mean()
        losses.update(loss.detach().cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        mre_errors.update(mre_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(out.detach().cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                  'MRE {mre_errors.val:.3f} ({mre_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors, mre_errors=mre_errors))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["cif_id", "target", "pred"])
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
        
        pred_t = torch.tensor(test_preds, dtype=torch.float32)
        targ_t = torch.tensor(test_targets, dtype=torch.float32)

        test_mae = float(torch.mean(torch.abs(pred_t - targ_t)))
        test_rmse = rmse(pred_t, targ_t)
        test_r2 = r2_score(pred_t, targ_t)
        
        if args.use_wandb:
            wandb.run.summary["test/mae"] = test_mae
            wandb.run.summary["test/rmse"] = test_rmse
            wandb.run.summary["test/r2"] = test_r2

            table = wandb.Table(columns=["cif_id", "target", "pred"])
            for cid, t, p in zip(test_cif_ids, test_targets, test_preds):
                table.add_data(cid, t, p)
            wandb.log({"test/results_table": table})
    else:
        star_label = '*'
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
        return float(losses.avg), float(mae_errors.avg)
        

def collect_pred_emb(args, loader, model, normalizer, device):
    model.eval()
    y_true, y_pred, emb_all = [], [], []
    cif_ids = []

    for i, (input, target, batch_cif_ids, xrd_fea, text_fea) in enumerate(loader):

        # ---- input_var ----
        if args.graph_type in ("cgcnn", "mpnn"):
            input_var = (
                input[0].to(device, non_blocking=True),
                input[1].to(device, non_blocking=True),
                input[2].to(device, non_blocking=True),
                [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]],
                xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None,
                text_fea.to(device, non_blocking=True) if text_fea is not None else None,
            )

            out, emb = model(*input_var)

        elif args.graph_type in ("chgnet", "m3gnet", "tensornet", "qet"):
            g, state_feats = input
            g = g.to(device)
            state_feats = state_feats.to(device, non_blocking=True)
            xrd_fea = xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None
            text_fea = text_fea.to(device, non_blocking=True) if text_fea is not None else None
            out, emb = model((g, state_feats), xrd_fea, text_fea)
        
        elif args.graph_type in ("alignn"):
            batch_g, batch_lg, batch_lat = input
            batch_g = batch_g.to(device)
            batch_lg = batch_lg.to(device)
            batch_lat = batch_lat.to(device)
            xrd_fea = xrd_fea.to(device, non_blocking=True) if xrd_fea is not None else None
            text_fea = text_fea.to(device, non_blocking=True) if text_fea is not None else None
            out, emb = model(batch_g, batch_lg, batch_lat, xrd_fea, text_fea)
        else:
            raise ValueError(f"Unknown graph_type: {args.graph_type}")

        # denorm prediction for metrics / plotting
        pred_denorm = normalizer.denorm(out.detach().cpu())
        y_pred.append(pred_denorm.view(-1))
        y_true.append(target.detach().cpu().view(-1))
        emb_all.append(emb.detach().cpu())
        cif_ids += batch_cif_ids

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    emb_all = torch.cat(emb_all).numpy()
    return y_true, y_pred, emb_all, cif_ids
    
    
def make_ood_umap_figure(train_emb, test_emb, y_true_test, y_pred_test, out_png,
                         density_threshold=1e-3, n_neighbors=50, min_dist=0.1):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score
    import umap

    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=0
    )
    all_emb = np.vstack([train_emb, test_emb])
    all_2d = reducer.fit_transform(all_emb)

    n_train = len(train_emb)
    train_2d = all_2d[:n_train]
    test_2d = all_2d[n_train:]

    kde = gaussian_kde(train_2d.T)
    test_density = kde(test_2d.T)

    in_mask = test_density >= density_threshold
    out_mask = ~in_mask

    r2_all = r2_score(y_true_test, y_pred_test)
    r2_in = r2_score(y_true_test[in_mask], y_pred_test[in_mask]) if in_mask.any() else float("nan")
    r2_out = r2_score(y_true_test[out_mask], y_pred_test[out_mask]) if out_mask.any() else float("nan")

    plt.figure(figsize=(6, 6))
    plt.scatter(train_2d[:, 0], train_2d[:, 1], s=6, alpha=0.5, c="gray", label="train")
    plt.scatter(test_2d[in_mask, 0], test_2d[in_mask, 1], s=10, alpha=0.9, c="blue", label="test in-domain")
    plt.scatter(test_2d[out_mask, 0], test_2d[out_mask, 1], s=10, alpha=0.9, c="red", label="test out-of-domain")

    plt.title(f"R2 all={r2_all:.3f} | in={r2_in:.3f} | out={r2_out:.3f}")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return {
        "r2_all": r2_all, "r2_in": r2_in, "r2_out": r2_out,
        "test_density": test_density,
        "in_mask": in_mask, "out_mask": out_mask,
        "train_2d": train_2d, "test_2d": test_2d,
    }


if __name__ == "__main__":
    args = arg_parse()
    
    # Check if Optuna distributed optimization flag is active
    if args.optuna_study:
        import optuna
        import copy
        
        def objective(trial):
            # 1. Clone original arguments to prevent trial pollution
            trial_args = copy.deepcopy(args)
            
            # 2. Define hyperparameter search space
            trial_args.batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
            trial_args.lr = trial.suggest_categorical('lr', [0.01, 0.001, 0.0001])
            trial_args.optim = trial.suggest_categorical('optim', ['Adam', 'SGD'])
            
            # GNN specific parameter (n_conv)
            if hasattr(trial_args, 'n_conv'):
                trial_args.n_conv = trial.suggest_int('n_conv', 3, 5)

            # Assign dynamic WandB run name based on parameters
            trial_args.wandb_name = f"optuna_t{trial.number}_bs{trial_args.batch_size}_lr{trial_args.lr}_{trial_args.optim}_nc{trial_args.n_conv}"
            
            # 3. Reset global best_mae_error to prevent evaluation carry-over
            global best_mae_error
            best_mae_error = 1e10
            
            # 4. Execute the main training loop with the suggested parameters
            val_error = main(trial_args)
            return val_error
            
        print(f"[Optuna Worker] DB: {args.optuna_study} | Study: {args.optuna_study_name}")
        
        # Join the distributed SQLite database optimization study
        study = optuna.load_study(
            study_name=args.optuna_study_name, 
            storage=args.optuna_study
        )
        study.optimize(objective, n_trials=args.optuna_n_trials)
    else:
        # Standard execution mode (Single run without Optuna)
        main(args)
