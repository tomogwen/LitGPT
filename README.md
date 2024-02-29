# Slurmformer

A minimal working example of multi-node, multi-GPU training using PyTorch Lightning with Wandb logging on a Slurm cluster. The model being trained is an implementation of Kaparthy's [minGPT](https://github.com/karpathy/minGPT).

This repo is aiming to grow into a (better than research code, aspiring to production quality) example of a deep learning repository - see the [main](https://github.com/tomogwen/LitGPT/tree/main) branch if interested.

## Usage

- Set the variables required in `train.sh` (your Slurm account details, required number of nodes and GPUs, and Wandb API key)
- Update the number of nodes and GPUs in `train.py` to match `train.sh`.
- Run `sbatch train.sh` on your Slurm cluster.
- Alternatively, run `python train.py` on any device.
