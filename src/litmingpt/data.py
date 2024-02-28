# Torch Dataset and LightningDataModule wrapper for the Tiny Shakespeare dataset

import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader


class TinyShakespeareDataSet(Dataset):
    def __init__(self, raw_text, block_size=256):
        super().__init__()
        self.raw_text = raw_text
        self.xs = torch.stack([raw_text[i:i+block_size] for i in range(len(raw_text) - block_size)])
        self.ys = torch.stack([raw_text[i+1:i+block_size+1] for i in range(len(raw_text) - block_size)])

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, index):
        return self.xs[index], self.ys[index]


class TinyShakespeareDataModule(L.LightningDataModule):
    def __init__(self, data_dir, train_dataloader_workers=10, val_dataloader_workers=10, batch_size=32, train_test_split=0.95):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.train_dataloader_workers = train_dataloader_workers
        self.val_dataloader_workers = val_dataloader_workers

    def prepare_data(self):
        # runs once, called from main process
        # tokenise data here
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            text = f.read()
            chars = sorted(list(set(text)))
        
            stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
            def encode(s):
                return [stoi[c] for c in s]  # encoder: maps strings to list of ints

            data = torch.tensor(encode(text), dtype=torch.long)
            torch.save(data, 'data/tokenised.pt')

    def setup(self, stage):
        # runs on every GPU
        # stage is e.g., "fit", "test"
        data = torch.load('data/tokenised.pt')

        n = int(self.train_test_split*len(data))
        self.train_data = TinyShakespeareDataSet(data[:n])
        self.val_data = TinyShakespeareDataSet(data[n:])

    def train_dataloader(self):
        # lightning should auto-add DistributedSampler for these dataloaders when required
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.train_dataloader_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.val_dataloader_workers, persistent_workers=True)
