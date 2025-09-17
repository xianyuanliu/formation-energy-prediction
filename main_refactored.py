"""Refactored main training script with modular architecture."""

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

from data import CIFData
from data import collate_pool, get_train_val_test_loader
from utils.utils import Normalizer, mae, save_checkpoint, AverageMeter
from config import ModelConfig, TrainerConfig, ModalityConfig, create_configs_from_args
from factories import model_factory

import warnings
warnings.filterwarnings("ignore", message=".*fractional coordinates rounded.*")


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')

    # Basic parameters
    parser.add_argument('--data_path', default='data/cifs', help='dataset path')
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

    # Model parameters
    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD', help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N', help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=128, type=int, metavar='N', help='number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N', help='number of conv layers')
    parser.add_argument('--n-h', default=1, type=int, metavar='N', help='number of hidden layers after pooling')
    parser.add_argument('--best_mae_error', default=1e10, type=float, metavar='N', help='best mae error (default: 1e10)')
    parser.add_argument('--graph_type', default="cgcnn", type=str, metavar="GRAPH", help='type of graph convolutional network (cgcnn or mpnn)')
    
    # Add model type for extensibility
    parser.add_argument('--model-type', default='cgcnn', type=str, help='model architecture type')
    
    args = parser.parse_args(sys.argv[1:])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    return args


def create_model_and_data(args, model_config, trainer_config, modality_config):
    """Create model and data loaders based on configurations."""
    
    # Load dataset
    dataset = CIFData(args.data_path, task=trainer_config.task)
    collate_fn = collate_pool
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=trainer_config.batch_size,
        train_ratio=trainer_config.train_ratio,
        train_size=trainer_config.train_size,
        val_ratio=trainer_config.val_ratio,
        val_size=trainer_config.val_size,
        test_ratio=trainer_config.test_ratio,
        test_size=trainer_config.test_size,
        return_test=True,
        num_workers=trainer_config.num_workers,
        pin_memory=trainer_config.cuda
    )

    # Get feature dimensions from dataset
    if trainer_config.task == 'regression':
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
    _, sample_target, _, _, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # Get data dimensions
    structures, _, _, _, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    # Create model using factory
    model = model_factory.create_model(
        model_config=model_config,
        modality_config=modality_config,
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len
    )
    
    if trainer_config.cuda:
        model.cuda()

    return model, train_loader, val_loader, test_loader, normalizer


def create_optimizer_and_scheduler(model, trainer_config):
    """Create optimizer and scheduler based on configuration."""
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Create optimizer
    if trainer_config.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), 
            trainer_config.learning_rate,
            momentum=trainer_config.momentum,
            weight_decay=trainer_config.weight_decay
        )
    elif trainer_config.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), 
            trainer_config.learning_rate,
            weight_decay=trainer_config.weight_decay
        )
    else:
        raise NameError('Only SGD or Adam is allowed as optimizer')
    
    # Create scheduler
    scheduler = MultiStepLR(optimizer, milestones=trainer_config.lr_milestones, gamma=0.1)
    
    return criterion, optimizer, scheduler


def main():
    # Parse arguments and create configurations
    args = arg_parse()
    model_config, trainer_config, modality_config = create_configs_from_args(args)
    
    # Set up CUDA
    trainer_config.cuda = args.cuda
    best_mae_error = trainer_config.best_mae_error

    print(f"Using model type: {model_config.model_type}")
    print(f"Available models: {model_factory.list_available_models()}")
    print(f"Using modalities - XRD: {modality_config.use_xrd}, Text: {modality_config.use_text}")

    # Create model and data
    model, train_loader, val_loader, test_loader, normalizer = create_model_and_data(
        args, model_config, trainer_config, modality_config
    )
    
    # Create optimizer and scheduler
    criterion, optimizer, scheduler = create_optimizer_and_scheduler(model, trainer_config)

    # Optionally resume from a checkpoint
    if trainer_config.resume:
        if os.path.isfile(trainer_config.resume):
            print("=> loading checkpoint '{}'".format(trainer_config.resume))
            checkpoint = torch.load(trainer_config.resume)
            trainer_config.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(trainer_config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(trainer_config.resume))

    # Training loop
    for epoch in range(trainer_config.start_epoch, trainer_config.epochs):
        # Train for one epoch
        train(trainer_config, train_loader, model, criterion, optimizer, epoch, normalizer)

        # Evaluate on validation set
        mae_error = validate(trainer_config, val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # Remember the best mae_error and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        
        # Save checkpoint with config info
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args),
            'model_config': model_config.__dict__,
            'trainer_config': trainer_config.__dict__,
            'modality_config': modality_config.__dict__
        }
        save_checkpoint(checkpoint_data, is_best)

    # Test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(trainer_config, test_loader, model, criterion, normalizer, test=True)


def train(config, train_loader, model, criterion, optimizer, epoch, normalizer):
    """Training function - now takes config object instead of args."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    mre_errors = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _, xrd_fea, text_fea) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        if config.cuda:
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
        # Normalize target
        target_normed = normalizer.norm(target)
        if config.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # Compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        mre_error = mae_error / target.abs().mean()

        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        mre_errors.update(mre_error, target.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                  'MRE {mre_errors.val:.3f} ({mre_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors, mre_errors=mre_errors)
            )


def validate(config, val_loader, model, criterion, normalizer, test=False):
    """Validation function - now takes config object instead of args."""
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    mre_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids, xrd_fea, text_fea) in enumerate(val_loader):
        if config.cuda:
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
        if config.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if config.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # Compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
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

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors))

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
    main()