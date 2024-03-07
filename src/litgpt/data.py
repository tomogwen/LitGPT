# Torch Dataset and LightningDataModule wrapper for Tiny Shakespeare dataset

import os

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class TinyShakespeareDataSet(Dataset):
    def __init__(self, raw_text, block_size=256):
        super().__init__()
        self.raw_text = raw_text
        self.xs = torch.stack(
            [raw_text[i : i + block_size] for i in range(len(raw_text) - block_size)]
        )
        self.ys = torch.stack(
            [
                raw_text[i + 1 : i + block_size + 1]
                for i in range(len(raw_text) - block_size)
            ]
        )

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]


class TinyShakespeareDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str = "data/tinyshakespeare.txt",
        batch_size: int = 64,
        train_test_split: float = 0.95,
        train_dataloader_workers: int = 10,
        val_dataloader_workers: int = 10,
        block_size: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()
        data_dir = os.path.dirname(self.hparams.dataset_path)
        self.hparams.tokenised_path = os.path.join(data_dir, "tokenised.pt")

    def prepare_data(self, tokenised_path=None):
        # runs once, called from main process
        # tokenise data here

        with open(self.hparams.dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
            chars = sorted(list(set(text)))

            stoi = {ch: i for i, ch in enumerate(chars)}  # string to int

            def encode(s):
                return [stoi[c] for c in s]  # encoder: maps strings to list of ints

            data = torch.tensor(encode(text), dtype=torch.long)
            torch.save(data, self.hparams.tokenised_path)

    def setup(self, stage):
        # runs on every GPU
        # stage is e.g., "fit", "test"
        data = torch.load(self.hparams.tokenised_path)

        n = int(self.hparams.train_test_split * len(data))
        self.train_data = TinyShakespeareDataSet(
            data[:n], block_size=self.hparams.block_size
        )
        self.val_data = TinyShakespeareDataSet(
            data[n:], block_size=self.hparams.block_size
        )

    def train_dataloader(self):
        # lightning should auto-add DistributedSampler for these dataloaders when required
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.train_dataloader_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.val_dataloader_workers,
            persistent_workers=True,
        )
