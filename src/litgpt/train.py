# Kaparthy's Min-GPT with a Lightning wrapper

import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from litgpt.data import TinyShakespeareDataModule
from litgpt.model import TransformerDecoder, LitMinGPT


def main():
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


if __name__ == '__main__':
    main()
