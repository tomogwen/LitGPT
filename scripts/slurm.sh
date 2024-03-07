#!/bin/bash
############################
# --  Set the following! --
############################
# --   Account Details   --
QOS = your_slurm_qos 
ACCOUNT = your_slurm_account
# --   Time Requested    -- 
hours = 1
mins = 0
seconds = 0
# -- Resources Requested --
NODES = 2
GPUS_PER_NODE = 2
CPUS_PER_GPU = 36
# --    Conda Setup      --
CONDA_ENVS_DIR = /path/to/envs/
CONDA_ENV_PATH = $CONDA_ENVS_DIR+="litgpt"
CONDA_PACKAGES_DIR = /path/to/packages/
############################
# If you want to use wanbd run
# > wanbd login
# to add creds to your .netrc
############################

#SBATCH --qos $QOS
#SBATCH --account $ACCOUNT
#SBATCH --time $H:$M:$S
#SBATCH --nodes $NODES
#SBATCH --gpus-per-node $GPUS_PER_NODE
#SBATCH --cpus-per-gpu $CPUS_PER_NODE
#SBATCH --ntasks-per-node $GPUS_PER_NODE

# Enable shell debugging
set -x

# Load conda
module purge
module load Miniconda3/4.10.3

# Setup conda
export CONDA_PKGS_DIRS=$CONDA_PACKAGES_DIR
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Install env if required
if [ ! -d "$CONDA_ENV_PATH" ]; then
    conda env create -f env.yml --prefix=$CONDA_ENV_PATH
fi

# Activate env
conda activate ${CONDA_ENVS_DIR}

# run train script
srun litgpt fit --config configs/slurm.yaml --trainer.devices $NODES --trainer.devices $GPUS_PER_NODE --data.train_dataloader_workers $CPUS_PER_NODE --data.val_dataloader_workers $CPUS_PER_NODE
