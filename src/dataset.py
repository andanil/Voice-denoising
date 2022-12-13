from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path


class ClassificationDataset(Dataset):
    def __init__(self, dataframe, data_dir, length):
        self.df = dataframe   
        self.data_dir = data_dir
        self.length = length
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):      
        spec = np.expand_dims(np.load(self.df.iloc[[index]]['path'].tolist()[0])[:self.length].T, axis=0)
        label = self.df.iloc[[index]]['class'].tolist()[0]
        
        return torch.as_tensor(spec, dtype=torch.float32), torch.as_tensor(label, dtype=torch.long)


class DenoisingDataset(Dataset):
    def __init__(self, dataframe, data_dir, length):
        self.df = dataframe   
        self.data_dir = data_dir
        self.length = length
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):      
        noisy = np.expand_dims(np.load(self.df.iloc[[index]]['noisy'].tolist()[0])[:self.length].T, axis=0)
        clean = np.expand_dims(np.load(self.df.iloc[[index]]['clean'].tolist()[0])[:self.length].T, axis=0)
        
        return torch.as_tensor(noisy, dtype=torch.float32), torch.as_tensor(clean, dtype=torch.float32)


def build_dataset(dataset_type, args):
    root = Path(args.dataset_folder)
    assert root.exists(), f'provided dataset path {root} does not exist'

    PATHS = {
        'train': (root / 'train', root / f"annotations_{'classif' if args.classify else 'denoising'}_train.json"),
        'val': (root / 'val', root / f"annotations_{'classif' if args.classify else 'denoising'}_val.json"),
    }

    folder, ann_file = PATHS[dataset_type]
    if args.classify:
        return ClassificationDataset(pd.read_json(ann_file, orient='records'), folder, args.input_dim)
    return DenoisingDataset(pd.read_json(ann_file, orient='records'), folder, args.input_dim)
