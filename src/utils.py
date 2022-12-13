import torch
import numpy as np

def save_model(*args, **kwargs):
    torch.save(*args, **kwargs)


def accuracy(y_pred, y_true):
    return torch.sum(torch.eq(y_true, torch.argmax(y_pred, dim=1)))/ y_true.shape[-1]

def load_file(args):
    return torch.as_tensor(np.load(args.input_file)[:args.input_dim].T[np.newaxis, np.newaxis, ...], dtype=torch.float32)

def save_to_file(prediction, output_file):
    np.save(output_file, prediction)