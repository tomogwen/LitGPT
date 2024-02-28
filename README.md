[![Pre-commit](https://github.com/tomogwen/LitGPT/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/tomogwen/LitGPT/actions/workflows/pre-commit.yml)&nbsp;&nbsp;[![Tests](https://github.com/tomogwen/LitGPT/actions/workflows/tests.yml/badge.svg)](https://github.com/tomogwen/LitGPT/actions/workflows/tests.yml)
# ⚡️ Lightning Minimal GPT

This repo contains my efforts to learn how to create a (better than research code, aspiring to production quality) deep learning repository. It trains an implementation of Kaparthy's [minGPT](https://github.com/karpathy/minGPT) in PyTorch Lightning.

## Goal

Some things that I'd like to learn whilst creating this repo are:

Software Development:
- [X] Setup github actions.
- [X] Writing tests.
- [X] Setup pre-commit checks.
- [X] 'Packagify' the code.
- [X] Good repo structure.

Deep Learning:
- [ ] Deal with hyperparams nicely
    - Config files + CLI
    - Use an args objects or pass around many hparams?
- [ ] Deal with different accelerators nicely
    - should run easily on CPU, MPS, or (multi-)GPU.
- [ ] Tune hyperparams
    - Understand the effect of tuning different hparams

I will hopefully add more to this as I go!

## Installation

To install dependencies:
```
> conda env create -f env.yml
```

Activate the conda environment:
```
> conda activate litgpt
```

To install pre-commit checks:
```
> pre-commit install
```

## Usage

To train the model locally (whilst in the conda environment):
```
> train
```

## Usage (slurm, outdated)

- Set the variables required in `train.sh` (your Slurm account details, required number of nodes and GPUs, and Wandb API key)
- Update the number of nodes and GPUs in `train.py` to match `train.sh`.
- Run `sbatch train.sh` on your Slurm cluster.
- Alternatively, run `python train.py` on any device.
