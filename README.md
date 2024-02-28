# Lightning GPT

This repo contains my efforts to learn how to create a (better than research code, not quite production quality) deep learning repository. It trains an implementation of Kaparthy's [minGPT](https://github.com/karpathy/minGPT) in PyTorch Lightning.

## Goal

Some things that I'd like to learn whilst creating this repo are: 

DevOps:
- [ ] Setup github actions, including automated pre-commits and CI.
    - How do actions work?
    - What do good tests look like for deep learning repos?
- [ ] Where should train scripts go?
- [X] 'Packagify' the code.
- [X] Have a conda env installer.

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
