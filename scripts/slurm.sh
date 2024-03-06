#!/bin/bash

#SBATCH --qos $QOS
#SBATCH --account $ACCOUNT
#SBATCH --time $H:$M:$S
#SBATCH --nodes $NODES
#SBATCH --gpus-per-node $GPUS_PER_NODE
#SBATCH --cpus-per-gpu $CPUS_PER_NODE
#SBATCH --ntasks-per-node $GPUS_PER_NODE

# Enable shell debugging
set -x

# Load modules if present on cluster, e.g.:
# module purge
# module load torchvision

# Set up venv
python -m venv --system-site-packages min-gpt-train
source min-gpt-train/bin/activate

# do pip installs
pip install torchvision
pip install lightning
pip install wandb

# init wandb
wandb login $WANDB_API_KEY

# run train script
srun litgpt fit --config configs/slurm.yaml --trainer.devices $NODES --trainer.devices $GPUS_PER_NODE --data.train_dataloader_workers $CPUS_PER_NODE --data.val_dataloader_workers $CPUS_PER_NODE
