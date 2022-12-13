import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from src.dataset import build_dataset
from src.engine import train, evaluate
from src.models.classifier import build_model as build_classifier
from src.models.dae import build_model as build_dae
import src.utils as utils


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Classifier parameters
    parser.add_argument('--num_classes', default=2, type=int)

    # DAE parameters
    parser.add_argument('--encoded_space_dim', default=512, type=int) # generally bigger than input size for DAE
    parser.add_argument('--input_dim', default=256, type=int)

    # other parameters
    parser.add_argument('--dataset_folder', default='./data')

    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--pretrained_path', type=str, default=None)
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='') # resume from checkpoint
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)

    return parser

def main(args):
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.classify:
        model, criterion = build_classifier(args)
        metric = utils.accuracy
    else:
        model, criterion = build_dae(args)
        metric = MeanSquaredError().to(device)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset_train = build_dataset(dataset_type='train', args=args)
    dataset_val = build_dataset(dataset_type='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)

    if args.pretrained_path is not None:
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        valid_epoch_loss, valid_epoch_metr = evaluate(model, data_loader_val, criterion, device, metric)
        print(f"Loss: {valid_epoch_loss:.3f}, {'acc' if args.classify else 'MSE'}: {valid_epoch_metr:.3f}")
        return

    print("Start training")

    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch {epoch+1} of {args.epochs}")

        train_epoch_loss, train_epoch_metr = train(model, data_loader_train, optimizer, criterion, device, metric, args.clip_max_norm)
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}_{'classif' if args.classify else 'dae'}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_model({'epoch': epoch,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'args': args,
                                }, checkpoint_path)

        valid_epoch_loss, valid_epoch_metr = evaluate(model, data_loader_val, criterion, device, metric)

        print(f"Training loss: {train_epoch_loss:.3f}, training {'acc' if args.classify else 'MSE'}: {train_epoch_metr:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation {'acc' if args.classify else 'MSE'}: {valid_epoch_metr:.3f}")
        print('-'*50)
        time.sleep(5)

    print('Training complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Models training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)