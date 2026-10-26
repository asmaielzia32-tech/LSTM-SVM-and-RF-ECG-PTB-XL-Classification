import os 
import torch
import numpy as np
import pandas as pd
import ast
import wfdb
from torch.utils.data import Dataset

# Human readable class labels for the PTB dataset
CLASSES = {
    "NORM": 0,
    "MI":   1,
    "STTC": 2,
    "CD":   3,
    "HYP":  4,
}

class PTBDataset(Dataset):

    def __init__(self, 
                 path: str, 
                 sampling_rate: int, 
                 df: pd.DataFrame,
                 mean: torch.Tensor = None, 
                 std: torch.Tensor = None ):
        
        self.path = path
        self.num_classes = len(CLASSES)

        self.filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'

        # Check if the column 'scp_codes' is already a dict or not
        df['scp_codes'] = df['scp_codes'].apply(
            lambda x: x if isinstance(x, dict) else ast.literal_eval(x)
        )
        
        stat_path = os.path.join(path, 'scp_statements.csv')
        stat_df = pd.read_csv(stat_path, index_col=0)
        #create a mask to filter only diagnostic classes
        stat_df = stat_df[stat_df['diagnostic'] == 1.0]

        def aggregate_diagnostic(y_dic):
            tmp = [] 
            for key in y_dic.keys():
                if key in stat_df.index:
                    tmp.append(stat_df.loc[key].diagnostic_class)
            return list(set(tmp))# Use a set to avoid duplicates

        df['diagnostic_superclass'] = df['scp_codes'].apply(aggregate_diagnostic)


        #index reset of the pandas dataframe 
        self.data = df.reset_index(drop=True)

        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.data)
    

    def __getitem__(self, idx) -> dict:
        sample = self.data.iloc[idx]

        record_path = os.path.join(self.path, sample[self.filename_col])
        # Return a tuple (signal,fields) where:
        # - signal is a numpy array of shape (n_samples, n_channels)
        # - fields is a dictionary containing the metadata of the record.
        signal, _ = wfdb.rdsamp(record_path)

        signal_tensor = torch.tensor(signal, dtype=torch.float32) 
        if self.mean is not None and self.std is not None:
            signal_tensor = (signal_tensor - self.mean) / (self.std + 1e-8) #avoid division by zero
        
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        
        superclasses = sample['diagnostic_superclass']

        for d_class in superclasses:
            if d_class in CLASSES:
                class_idx = CLASSES[d_class]
                label_tensor[class_idx] = 1.0

        return {
            'signal': signal_tensor, # DIM (1000,12) OR (5000,12)
            'label': label_tensor # DIM 5
        }
