# Training minGPT with Lightningloss

import torch
from lightning.pytorch.cli import LightningCLI

from litgpt.data import TinyShakespeareDataModule
from litgpt.model import LitMinGPT


def main():
    torch.set_float32_matmul_precision("high")
    LightningCLI(LitMinGPT, TinyShakespeareDataModule)


if __name__ == "__main__":
    main()
