[![Pre-commit](https://github.com/tomogwen/LitGPT/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/tomogwen/LitGPT/actions/workflows/pre-commit.yml)&nbsp;&nbsp;[![Tests](https://github.com/tomogwen/LitGPT/actions/workflows/tests.yml/badge.svg)](https://github.com/tomogwen/LitGPT/actions/workflows/tests.yml)
# ⚡️ Lightning Minimal GPT

This repo contains my efforts to learn how to create a (better than research code, aspiring to production quality) deep learning repository. It trains an implementation of Kaparthy's [minGPT](https://github.com/karpathy/minGPT) in PyTorch Lightning.

**MWE:** The code here grew out of a minimal example of multi-node, multi-GPU training with PyTorch Lightning on a slurm cluster - if you're interested in that, please see the [slurmformer branch](https://github.com/tomogwen/LitGPT/tree/slurmformer).

## Goal

A non-exhaustive list of skills I'd like to learn about via this repo are listed below.

Machine Learning Engineering:
- [ ] Dealing with hyperparams nicely
    - Config files + CLI
    - Use an args objects or pass around many hparams?
- [ ] Dealing with different accelerators nicely
    - should run easily on CPU, MPS, or (multi-)GPU.

Software development:
- [ ] Doc strings and type hints
- [X] Setting up github actions.
- [X] Writing tests.
- [X] Setting up pre-commit checks.
- [X] 'Packagify'-ing code.
- [X] Having good repo structure.

## Installation

To install dependencies and activate the conda environment:
```
> conda env create -f env.yml
> conda activate litgpt
```

If developing, install pre-commit checks:
```
> pre-commit install
```

## Usage

To train the model locally (whilst in the conda environment):
```
> train
```
