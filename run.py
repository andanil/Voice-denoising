import argparse
import torch
import numpy as np

from src.engine import predict
from src.models.classifier import build_model as build_classifier
from src.models.dae import build_model as build_dae
import src.utils as utils


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Classifier parameters
    parser.add_argument('--num_classes', default=2, type=int)

    # DAE parameters
    parser.add_argument('--encoded_space_dim', default=512, type=int) # generally bigger than input size for DAE
    parser.add_argument('--input_dim', default=256, type=int)

    # other parameters
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)

    parser.add_argument('--pretrained_path', type=str, default=None)
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)

    return parser


def main(args):
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    
    if args.classify:
        model, _ = build_classifier(args)
    else:
        model, _ = build_dae(args)

    model.to(device)

    assert args.input_file is not None, 'Input file must be specified'
    data = utils.load_file(args)

    assert args.pretrained_path is not None, 'Model must be specified'
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    result = predict(model, data, device)

    if args.classify:
        print(f"Predicted class: {'noisy' if np.argmax(result) == 1 else 'clean'}")
        return

    print('Prediction complete')
    if args.output_file is not None:
        utils.save_to_file(result, args.output_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mel spectrograms classification and denoising script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)