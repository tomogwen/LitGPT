import os

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TinyShakespeareDataSet(Dataset):
    """
    Torch Dataset for the Tiny Shakespeare dataset.
    """

    def __init__(self, raw_text: Tensor, block_size: int = 256):
        """
        Args:
            raw_text: Tensor of tokens, processed in TinyShakespeareDataModule.
            block_size: number of tokens in each training sample.
        """
        super().__init__()
        self.raw_text: Tensor = raw_text
        self.xs: Tensor = torch.stack(
            [raw_text[i : i + block_size] for i in range(len(raw_text) - block_size)]
        )
        self.ys: Tensor = torch.stack(
            [
                raw_text[i + 1 : i + block_size + 1]
                for i in range(len(raw_text) - block_size)
            ]
        )

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, index) -> int:
        return self.xs[index], self.ys[index]


class TinyShakespeareDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the Tiny Shakespeare dataset.
    """

    def __init__(
        self,
        dataset_path: str = "data/tinyshakespeare.txt",
        batch_size: int = 64,
        train_test_split: float = 0.95,
        train_dataloader_workers: int = 10,
        val_dataloader_workers: int = 10,
        block_size: int = 256,
    ):
        """
        Args:
            dataset_path: path to a text file (typically) containing the complete works of Shakespeare.
            batch_size: number of datapoints given on each sample from the dataset.
            train_test_split: number in [0, 1] representing what proportion of the data is used as training data.
            train_dataloader_workers: passed to the train dataloader num_workers arg.
            val_dataloader_workers: passed to the val dataloader num_workers arg.
            block_size: number of tokens in each training sample.
        """
        super().__init__()
        self.save_hyperparameters()
        data_dir: str = os.path.dirname(self.hparams.dataset_path)
        self.hparams.tokenised_path: str = os.path.join(data_dir, "tokenised.pt")

    def prepare_data(self, tokenised_path=None):
        """Loads data from txt file, tokenises it, then saves as a Tensor. Runs once from parent process."""
        with open(self.hparams.dataset_path, "r", encoding="utf-8") as f:
            text: str = f.read()
            chars: list[str] = sorted(list(set(text)))

            # tokeniser
            stoi: dict = {ch: i for i, ch in enumerate(chars)}  # string to int

            def encode(s: str) -> list[int]:
                return [stoi[c] for c in s]  # encoder: maps strings to list of ints

            # tokenise data
            data: torch.tensor = torch.tensor(encode(text), dtype=torch.long)
            torch.save(data, self.hparams.tokenised_path)

    def setup(self, stage):
        """Loads Tensor of Tokens and splits into train/val datasets. Runs on each GPU if using DDP."""
        data: torch.tensor = torch.load(self.hparams.tokenised_path)

        n: int = int(self.hparams.train_test_split * len(data))
        self.train_data: TinyShakespeareDataSet = TinyShakespeareDataSet(
            data[:n], block_size=self.hparams.block_size
        )
        self.val_data: TinyShakespeareDataSet = TinyShakespeareDataSet(
            data[n:], block_size=self.hparams.block_size
        )

    def train_dataloader(self) -> DataLoader:
        """Returns a dataloader for the training dataset."""
        # lightning auto-adds DistributedSampler for these dataloaders when required
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.train_dataloader_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a dataloader for the validation dataset."""
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.val_dataloader_workers,
            persistent_workers=True,
        )
