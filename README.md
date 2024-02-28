# Lightning MinGPT

This repo contains my efforts to learn how to train models with PyTorch Lightning. It currently has a minimal working example of multi-node, multi-GPU training for Kaparthy's [minGPT](https://github.com/karpathy/minGPT) on a Slurm cluster.

## Goal

I'd ultimately like this repo to be an example of a production deep learning repo following best practices.

The following are (I believe) devops'y questions with that goal in mind:
- [ ] Setup github actions, including automated pre-commits and CI.
    - How do actions work?
    - What do good tests look like for deep learning repos?
- [ ] Where should train scripts go?
- [X] 'Packagify' the code.
- [X] Have a conda env installer.

The following are more about the deep learning skills for a production quality repo:
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

## Usage

To train the model locally:
```
> conda activate litmingpt
> train
```

## Usage (slurm, outdated)

- Set the variables required in `train.sh` (your Slurm account details, required number of nodes and GPUs, and Wandb API key)
- Update the number of nodes and GPUs in `train.py` to match `train.sh`.
- Run `sbatch train.sh` on your Slurm cluster.
- Alternatively, run `python train.py` on any device.
