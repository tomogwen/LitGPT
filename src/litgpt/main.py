# Minimal GPT Implementation in PyTorch Lightning
# https://github.com/tomogwen/litgpt

import torch
from lightning.pytorch.cli import LightningCLI

from litgpt.data import TinyShakespeareDataModule
from litgpt.model import LitMinGPT


def main():
    """LightningCLI entry point."""
    torch.set_float32_matmul_precision("high")
    LightningCLI(LitMinGPT, TinyShakespeareDataModule, save_config_callback=None)


if __name__ == "__main__":
    main()
