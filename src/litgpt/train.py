# Training minGPT with Lightningloss

import lightning as L
import torch
from pytorch_lightning.loggers import WandbLogger

from litgpt.data import TinyShakespeareDataModule
from litgpt.model import LitMinGPT


def get_hparams():
    # fmt: off
    return {
        # Program args
        "dataset_path": "data/tinyshakespeare.txt",
        "accelerator": "mps",
        "logging": None,
        "train_dataloader_workers": 10,
        "val_dataloader_workers": 10,

        # Model args
        "VOCAB_SIZE": 65,
        "N_EMBD": 384,  # dimension of token embeddings
        "N_HEADS": 6,  # number of self-attention heads
        "NUM_BLOCKS": 3,  # number of transformer blocks
        "BATCH_SIZE": 64,  # how many independent sequences processed in paralell?
        "BLOCK_SIZE": 256,  # maximum context length for the transformer (max T)
        "DROPOUT": 0.2,  # propo of dropout

        # Trainer args
        "max_epochs": 10,
        "optimiser_name": "adam",
        "lr": 3e-4,
        "scheduler_name": None,
        "batch_size": 32,
        "train_test_split": 0.95,
    }  # fmt: on


def main():
    # hparams
    hparams = get_hparams()

    # data module
    # data_module = TinyShakespeareDataModule("data/tinyshakespeare.txt")
    data_module = TinyShakespeareDataModule(hparams)
    data_module.prepare_data()
    data_module.setup("train")

    # init model
    litgpt = LitMinGPT(hparams)

    if litgpt.hparams.logging == "wandb":
        # weights and biases
        wandb_logger = WandbLogger(log_model="all", project="hpc-gpt")
        logger = wandb_logger
    else:
        logger = None

    if litgpt.hparams.accelerator == "gpu":
        # drop f32 precision for tensor cores
        torch.set_float32_matmul_precision("high")

        # train model
        num_devices = 2
        gpus_per_device = 2

        trainer = L.Trainer(
            accelerator="gpu",
            devices=num_devices,
            num_nodes=gpus_per_device,
            strategy="ddp",
            logger=logger,
            max_epochs=10,
        )
    else:
        # auto-choose accelerator
        trainer = L.Trainer(logger=logger, max_epochs=10)

    trainer.fit(model=litgpt, datamodule=data_module)


if __name__ == "__main__":
    main()
