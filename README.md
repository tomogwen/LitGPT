[![Pre-commit](https://github.com/tomogwen/LitGPT/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/tomogwen/LitGPT/actions/workflows/pre-commit.yml)&nbsp;&nbsp;[![Tests](https://github.com/tomogwen/LitGPT/actions/workflows/tests.yml/badge.svg)](https://github.com/tomogwen/LitGPT/actions/workflows/tests.yml)
# ⚡️ Lightning Minimal GPT

This repo trains a PyTorch implementation of [minGPT](https://github.com/karpathy/minGPT) using PyTorch Lightning. MinGPT is a minimal version of a [GPT language model](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) as taught in Kaparthy's [zero-to-hero course](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy). This codebase is a 'playground' repository where I can practice writing (hopefully!) better deep learning code.

## 🔧 Installation

To install dependencies and activate the conda environment:
```
conda env create -f env.yml
conda activate litgpt
```

If developing, install pre-commit checks:
```
pre-commit install
```

## 📈 Training

To train the model (whilst in the conda environment):
```
litgpt fit --config configs/default.yaml
```

You can override and extend the config file using the CLI. Arguments like `--optimizer` and `--lr_scheduler` accept Torch classes. Run `litgpt fit --help` or read the [LightningCLI docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for all options.


### 👀 Logging

We provide config files for [Tensorboard](https://www.tensorflow.org/tensorboard) and [Weights & Biases](https://wandb.ai/) monitoring. Training with the default config (as above) uses Tensorboard. You can monitor training by running:

```
tensorboard --log-dir=checkpoints/
```

To log with Weights & Biases use the `default_wandb.yaml` or `ddp.yaml` config files. You will need to authenticate for the first time using `wandb login`.

### 🚀 HPC

A script for [DDP training](https://pytorch.org/tutorials/beginner/ddp_series_theory.html) on Slurm-managed HPC is provided. Update the [shell script](scripts/slurm.sh) where required, make it executable (with `chmod +x scripts/slurm.sh`), and run it:
```
scripts/slurm.sh
```
This script will generate and submit a slurm job using `sbatch`. Generating the script dynamically allows resource requests to be set once at the top of the file, then passed to both slurm (to allocate resources) and Lightning (to utilise them).
