"""Refactored test script with modular architecture."""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import CIFData
from data import collate_pool
from utils.utils import Normalizer, mae, AverageMeter
from config import ModelConfig, TrainerConfig, ModalityConfig
from factories import model_factory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crystal Graph Neural Networks Testing')
    parser.add_argument('--modelpath', help='path to the trained model.', 
                       default="pretrained_models/formation-energy-per-atom.pth")
    parser.add_argument('--cifpath', help='path to the directory of cifs files.', 
                       default="data/test_cifs/")
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                       metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                       help='number of data loading workers (default: 0)')
    parser.add_argument('--disable-cuda', action='store_true',
                       help='Disable CUDA')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args(sys.argv[1:])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    return args


def load_model_from_checkpoint(modelpath, cuda_available=True):
    """Load model configurations and state from checkpoint."""
    
    if not os.path.isfile(modelpath):
        raise FileNotFoundError(f"=> no model params found at '{modelpath}'")
        
    print(f"=> loading model params '{modelpath}'")
    model_checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
    
    # Extract configurations (backward compatibility)
    if 'model_config' in model_checkpoint:
        # New format with explicit configs
        model_config_dict = model_checkpoint['model_config']
        trainer_config_dict = model_checkpoint['trainer_config'] 
        modality_config_dict = model_checkpoint['modality_config']
        
        model_config = ModelConfig(**model_config_dict)
        trainer_config = TrainerConfig(**trainer_config_dict)
        modality_config = ModalityConfig(**modality_config_dict)
    else:
        # Old format - extract from args
        old_args = argparse.Namespace(**model_checkpoint['args'])
        model_config = ModelConfig(
            model_type=getattr(old_args, 'model_type', 'cgcnn'),
            atom_fea_len=getattr(old_args, 'atom_fea_len', 64),
            n_conv=getattr(old_args, 'n_conv', 3),
            h_fea_len=getattr(old_args, 'h_fea_len', 128),
            n_h=getattr(old_args, 'n_h', 1),
            graph_type=getattr(old_args, 'graph_type', 'cgcnn')
        )
        trainer_config = TrainerConfig(
            task=getattr(old_args, 'task', 'regression'),
            cuda=cuda_available
        )
        modality_config = ModalityConfig(
            use_xrd=getattr(old_args, 'xrd', False),
            use_text=getattr(old_args, 'text', False)
        )
    
    print(f"=> loaded model params '{modelpath}'")
    return model_checkpoint, model_config, trainer_config, modality_config


def create_test_model(model_config, modality_config, dataset):
    """Create model for testing based on configurations and dataset."""
    
    # Get feature dimensions from dataset
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    
    # Create model using factory
    model = model_factory.create_model(
        model_config=model_config,
        modality_config=modality_config,
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len
    )
    
    return model


def main():
    # Parse arguments
    args = parse_args()

    # Load model from checkpoint
    try:
        model_checkpoint, model_config, trainer_config, modality_config = load_model_from_checkpoint(
            args.modelpath, args.cuda
        )
    except FileNotFoundError as e:
        print(e)
        return

    # Set up task-specific variables
    if trainer_config.task == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.

    print(f"Testing with model type: {model_config.model_type}")
    print(f"Task: {trainer_config.task}")
    print(f"Modalities - XRD: {modality_config.use_xrd}, Text: {modality_config.use_text}")

    # Load test data
    dataset = CIFData(args.cifpath)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, collate_fn=collate_fn,
                           pin_memory=args.cuda)

    # Create model
    model = create_test_model(model_config, modality_config, dataset)
    
    if args.cuda:
        model.cuda()

    # Load model state
    model.load_state_dict(model_checkpoint['state_dict'])

    # Define loss function and normalizer
    criterion = nn.MSELoss()
    normalizer = Normalizer(torch.zeros(3))
    
    if 'normalizer' in model_checkpoint:
        normalizer.load_state_dict(model_checkpoint['normalizer'])

    # Run testing
    print("Running model evaluation...")
    validate(trainer_config, test_loader, model, criterion, normalizer, test=True)


def validate(config, val_loader, model, criterion, normalizer, test=False):
    """Validation function using configuration object."""
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
        if config.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        else:
            mae_error = class_eval(output.data.cpu(), target)
        
        mre_error = mae_error / target.abs().mean() if config.task == 'regression' else 0
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        mre_errors.update(mre_error, target.size(0))
        
        if test:
            if config.task == 'regression':
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
            else:
                test_pred = torch.softmax(output.data.cpu(), dim=1)
                test_target = target
                
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % getattr(config, 'print_freq', 10) == 0:
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


def class_eval(prediction, target):
    """Evaluate classification task."""
    prediction = np.exp(prediction.numpy())
    prediction = np.argmax(prediction, axis=1)
    target = target.view(-1).numpy()
    return metrics.accuracy_score(target, prediction)


if __name__ == '__main__':
    main()