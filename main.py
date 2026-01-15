import sys, types
sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))

# ---- torchdata.datapipes compatibility shim (must be before importing dgl) ----
import sys, types
from torch.utils.data.datapipes.datapipe import IterDataPipe  # ✅ correct location for your torch

pkg = types.ModuleType("torchdata.datapipes")
pkg.__path__ = []  # mark as package-like
sys.modules.setdefault("torchdata.datapipes", pkg)

mod = types.ModuleType("torchdata.datapipes.iter")
mod.IterDataPipe = IterDataPipe
sys.modules["torchdata.datapipes.iter"] = mod
# -----------------------------------------------------------------------------

import argparse
import os
import sys
import time
from random import sample

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data import CIFData
from data import collate_pool, get_train_val_test_loader, collate_pool_matgl
from models.cgcnn import CrystalGraphConvNet
from models.cgcnn import MatglGraphConvNet
from utils.utils import Normalizer, mae, save_checkpoint, AverageMeter
from thop import profile
import datetime
import wandb

import warnings
warnings.filterwarnings("ignore", message=".*fractional coordinates rounded.*")

def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')

    # Basic parameters
    parser.add_argument('--data_path', default='data/split_both_hhi', help='path to csv files')
    parser.add_argument('--cif_path', default='data/cifs', help='path to cif files')
    parser.add_argument('--task', default='regression')
    parser.add_argument('--xrd', default=True, help='use xrd features')
    parser.add_argument('--text', default=True, help='use text features')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run (default: 30)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int, metavar='N', help='milestones for scheduler (default: [100])')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_file', default=None, help='train csv file name in data_path')
    parser.add_argument('--test_file', default=None, help='test csv file name in data_path')
    # WandB parameters
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB for logging')
    parser.add_argument('--wandb_project', default='formatin-energy-preiction-project', type=str, help='WandB project name')
    parser.add_argument('--wandb_group', default='baseline', type=str, help='WandB group name')
    parser.add_argument('--wandb_name', default=None, type=str, help='WandB run name (None = auto-generated)')

    # Data split
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('--train-ratio', default=None, type=float, metavar='N', help='number of training data to be loaded (default none)')
    train_group.add_argument('--train-size', default=None, type=int, metavar='N', help='number of training data to be loaded (default none)')

    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N', help='percentage of validation data to be loaded (default 0.1)')
    valid_group.add_argument('--val-size', default=None, type=int, metavar='N', help='number of validation data to be loaded (default 1000)')

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N', help='percentage of test data to be loaded (default 0.1)')
    test_group.add_argument('--test-size', default=None, type=int, metavar='N', help='number of test data to be loaded (default 1000)')

    # model parameters
    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD', help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N', help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=128, type=int, metavar='N', help='number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N', help='number of conv layers')
    parser.add_argument('--n-h', default=1, type=int, metavar='N', help='number of hidden layers after pooling')
    parser.add_argument('--best_mae_error', default=1e10, type=float, metavar='N', help='best mae error (default: 1e10)')
    parser.add_argument('--graph_type', default="cgcnn", type=str, metavar="GRAPH", help='type of graph convolutional network (cgcnn or mpnn)')
    args = parser.parse_args(sys.argv[1:])
    return args

best_mae_error = 1e10

def main():
    global best_mae_error
    args = arg_parse()
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
            mode="offline",
            settings=wandb.Settings(console="off")
        )
    start_total_time = time.time()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.graph_type in ("cgcnn", "mpnn"):
        collate_fn = collate_pool
    elif args.graph_type in ("chgnet", "m3gnet"):
        collate_fn = collate_pool_matgl

    # Data loader generation (Conditional branching)
    if args.train_file and args.test_file:
        # Mode A: Use separate files for training and testing
        print(f"=> Separate file mode: {args.train_file} (train) / {args.test_file} (test)")
        
        # Load full training dataset
        full_train_dataset = CIFData(args.data_path, cif_path=args.cif_path, csv_filename=args.train_file, graph_type=args.graph_type)
        # Load test dataset
        test_dataset = CIFData(args.data_path, cif_path=args.cif_path, csv_filename=args.test_file, graph_type=args.graph_type)
        
        # Calculate indices for validation split from the train file
        indices = list(range(len(full_train_dataset)))
        val_size = int(len(full_train_dataset) * args.val_ratio)
        train_size = len(full_train_dataset) - val_size
        
        # Set up samplers for random split
        train_sampler = SubsetRandomSampler(indices[:train_size])
        val_sampler = SubsetRandomSampler(indices[train_size:])
        
        # Create DataLoaders
        train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=args.workers,
                                  collate_fn=collate_fn, pin_memory=args.cuda)
        
        val_loader = DataLoader(full_train_dataset, batch_size=args.batch_size,
                                sampler=val_sampler, num_workers=args.workers,
                                collate_fn=collate_fn, pin_memory=args.cuda)
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers,
                                 collate_fn=collate_fn, pin_memory=args.cuda)
        
        # Set representative dataset for model building
        dataset = full_train_dataset

    else:
        # Mode B: Original behavior (Split single file by ratios)
        print("=> Combined file mode: Using 1_MatDX_EF_modified.csv with ratio split")
        dataset = CIFData(args.data_path, graph_type=args.graph_type)

        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
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
        
    elif args.graph_type in ("chgnet", "m3gnet"):
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
    if args.cuda:
        model.cuda()

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
        else:
        # MatglGraphConvNet (graph_state, xrd, text)
            graph_state = (
                sample_data[0][0].to(device), # batch_graph
                sample_data[0][1].to(device)  # state_feats
            )
            inputs = (
                graph_state,
                sample_data[3].to(device),    # xrd_feature
                sample_data[4].to(device)     # text_feature
            )

        flops, params = profile(model, inputs=inputs, verbose=False)
        if args.use_wandb:
            wandb.config.update({
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
    criterion = nn.MSELoss()
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

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(args, val_loader, model, criterion, normalizer)
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "val/mae": mae_error,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(args, test_loader, model, criterion, normalizer, test=True)
    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    total_time_str = str(datetime.timedelta(seconds=int(total_duration)))
    if args.use_wandb:
        wandb.run.summary["total_training_time_sec"] = total_duration
        wandb.run.summary["best_mae_error"] = best_mae_error
        wandb.finish() # 프로세스 종료

    print("\n" + "="*30)
    print(f"  Training Completed!")
    print(f"  Total Time Elapsed: {total_time_str}")
    print(f"  ({total_duration:.2f} seconds)")
    print("="*30)


def train(args, train_loader, model, criterion, optimizer, epoch, normalizer):
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
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                            Variable(input[1].cuda(non_blocking=True)),
                            input[2].cuda(non_blocking=True),
                            [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                            xrd_fea.cuda(non_blocking=True) if xrd_fea is not None else None,
                            text_fea.cuda(non_blocking=True) if text_fea is not None else None)
            else:
                input_var = (Variable(input[0]),
                            Variable(input[1]),
                            input[2],
                            input[3],
                            xrd_fea,
                            text_fea)
        elif args.graph_type in ("chgnet", "m3gnet"):
            batch_graph, batch_state = input

            if args.cuda:
                batch_graph = batch_graph.to("cuda") 
                batch_state = batch_state.cuda(non_blocking=True)
                xrd_fea = xrd_fea.cuda(non_blocking=True) if xrd_fea is not None else None
                text_fea = text_fea.cuda(non_blocking=True) if text_fea is not None else None
            input_var = ((batch_graph, batch_state), xrd_fea, text_fea)

        else:
            raise ValueError(f"Unknown graph_type: {args.graph_type}")
        
        target_normed = normalizer.norm(target)
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        if args.graph_type in ("cgcnn", "mpnn"):
            output = model(*input_var)
        elif args.graph_type in ("chgnet", "m3gnet"):
            # input_var = (graph_state, xrd_fea, text_fea)
            graph_state, xrd_in, text_in = input_var
            output = model(graph_state, xrd_in, text_in)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        mre_error = mae_error / target.abs().mean()

        losses.update(loss.data.cpu(), target.size(0))
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
            if args.use_wandb:
                wandb.log({
                    "train/batch_loss": losses.val,
                    "train/batch_mae": mae_errors.val
                })
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                  'MRE {mre_errors.val:.3f} ({mre_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors, mre_errors=mre_errors)
            )


def validate(args, val_loader, model, criterion, normalizer, test=False):
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
            if args.cuda:
                with torch.no_grad():
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                Variable(input[1].cuda(non_blocking=True)),
                                input[2].cuda(non_blocking=True),
                                [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                                xrd_fea.cuda(non_blocking=True) if xrd_fea is not None else None,
                                text_fea.cuda(non_blocking=True) if text_fea is not None else None)
            else:
                with torch.no_grad():
                    input_var = (Variable(input[0]),
                                Variable(input[1]),
                                input[2],
                                input[3],
                                xrd_fea,
                                text_fea)
        elif args.graph_type in ("chgnet", "m3gnet"):
            graph_state = input 
            if args.cuda:
                with torch.no_grad():
                    g, state_feats = graph_state
                    g = g.to("cuda")
                    state_feats = state_feats.cuda(non_blocking=True)
                    xrd_fea_cuda = xrd_fea.cuda(non_blocking=True) if xrd_fea is not None else None
                    text_fea_cuda = text_fea.cuda(non_blocking=True) if text_fea is not None else None
                    input_var = ((g, state_feats), xrd_fea_cuda, text_fea_cuda)
            else:
                with torch.no_grad():
                    input_var = (graph_state, xrd_fea, text_fea)
        else:
            raise ValueError(f"Unknown graph_type: {args.graph_type}")

        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        mre_error = mae_error / target.abs().mean()
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        mre_errors.update(mre_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
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
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
        return mae_errors.avg


if __name__ == '__main__':
    # --- Configuration for testing/running the script ---
    
    # Mode A: Individual File Mode
    # Use this if you have physically separated train.csv and test.csv files.
    # Validation data will be automatically split from the train.csv based on --val-ratio.
    sys.argv += [
        '--graph_type', 'chgnet', 
        '--data_path', 'data/split_both_hhi',
        '--train_file', 'train.csv', 
        '--test_file', 'test.csv',
        '--use_wandb',
        '--wandb_group', 'CHGNet-Baseline', 
        '--wandb_name', 'hhi'
    ]
    
    # Mode B: Combined File Mode (Original Behavior)
    # Use this to load the default '1_MatDX_EF_modified.csv' and split it by ratios.
    # To use this mode, comment out Mode 1 above and uncomment the line below.
    #sys.argv += [
    #    '--graph_type', 'chgnet'] 
    #    ]
    
    main()
