# Kaparthy's Min-GPT with a Lightning wrapper

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from MinGPT import TransformerDecoder


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


class LitMinGPT(L.LightningModule):
    def __init__(self, transformer_decoder):
        super().__init__()
        # self.save_hyperparameters()  # don't need to specify for any nn.Module
        self.decoder = transformer_decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.decoder(x, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.decoder(x, y)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
         optimiser = torch.optim.AdamW(self.parameters(), lr=1e-3)
         return optimiser


if __name__ == '__main__':
    
    accelerator = "mps"
    logging = None

    # data module
    data_module = TinyShakespeareDataModule("data/tinyshakespeare.txt")
    data_module.prepare_data()
    data_module.setup('train')

    # init model
    model = TransformerDecoder()
    litgpt = LitMinGPT(model)

    if logging == 'wandb':
        # weights and biases
        wandb_logger = WandbLogger(log_model="all", project="hpc-gpt")
        logger = wandb_logger
    else:
        logger = None

    if accelerator == "gpu":
        # drop f32 precision for tensor cores
        torch.set_float32_matmul_precision('high')
        
        # train model
        num_devices = 2
        gpus_per_device = 2 

        trainer = L.Trainer(accelerator="gpu", devices=num_devices, num_nodes=gpus_per_device, strategy="ddp", logger=logger, max_epochs=10)
    else:
        # auto-choose accelerator
        trainer = L.Trainer(logger=logger, max_epochs=10)

    trainer.fit(model=litgpt, datamodule=data_module)
