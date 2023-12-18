# Slurmformer

A minimal working example of multi-node, multi-GPU training using PyTorch Lightning with Wandb logging on a Slurm cluster. The model being trained is an implementation of Kaparthy's [min-GPT](https://github.com/karpathy/minGPT).

## Usage

- Set the variables required in `train.sh` (your Slurm account details, required number of nodes and GPUs, and Wandb API key)
- Run `sbatch train.sh` on your Slurm cluster.
- Alternatively, run `python train.py` on any device.

