# Slurmformer

This repo contains my efforts to learn how to train models with PyTorch Lightning. It currently has a minimal working example of multi-node, multi-GPU training for Kaparthy's [minGPT](https://github.com/karpathy/minGPT) on a Slurm cluster.

## Goal

I'd ultimately like this repo to be an example of a production deep learning repo following best practices.

The following are (I believe) devops'y questions with that goal in mind:
- [ ] Have a directory structure that represents best practices.
    - What should be top-level, what should be in `src`?
    - If e.g., `data` is top-level and `train.py` is in `src`, how does `train.py` see the data?
- [ ] Have a conda env installer, including installing a python package locally.
    - How does it work to be able to `pip install -e .`?
    - What does `setup.py` and `pyproject.toml` do?
- [ ] Setup github actions, including automated pre-commits and CI.
    - How do actions work?
    - What should be in `.pre-commit-config.yaml`?
    - What do good tests look like for deep learning repos?

The following are more about the deep learning skills for a production quality repo:
- [ ] Deal with hyperparams nicely
    - Config files + CLI
    - Use an args objects or pass around many hparams?
- [ ] Tune hyperparams
    - Understand the effect of tuning different hparams

I will hopefully add more to this as I go!

## Installation

To install dependencies in a conda environment and activitate it run the following.

```
> conda env create -f env.yml
> conda activate slurmformers
```

## Usage (slurm, outdated)

- Set the variables required in `train.sh` (your Slurm account details, required number of nodes and GPUs, and Wandb API key)
- Update the number of nodes and GPUs in `train.py` to match `train.sh`.
- Run `sbatch train.sh` on your Slurm cluster.
- Alternatively, run `python train.py` on any device.
