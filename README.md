[![Pre-commit](https://github.com/tomogwen/LitGPT/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/tomogwen/LitGPT/actions/workflows/pre-commit.yml)&nbsp;&nbsp;[![Tests](https://github.com/tomogwen/LitGPT/actions/workflows/tests.yml/badge.svg)](https://github.com/tomogwen/LitGPT/actions/workflows/tests.yml)
# ⚡️ Lightning Minimal GPT

This repo contains my efforts to learn how to create a (better than research code, aspiring to production quality) deep learning repository. It trains an implementation of Kaparthy's [minGPT](https://github.com/karpathy/minGPT) in PyTorch Lightning.

**MWE:** The code here grew from a minimal example of distributed training on a slurm cluster. If you're interested in that, please see the [slurmformer branch](https://github.com/tomogwen/LitGPT/tree/slurmformer).

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

You can override and extend the config file using the CLI. Arguments like `--optimizer` and `--lr_scheduler` accept Torch classes. For example:
```
litgpt fit --config configs/default.yaml --optimizer Adam
```

This uses the [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html#). All options can be seen by running `litgpt fit --help`.

### 🚀 HPC

A script for [DDP training](https://pytorch.org/tutorials/beginner/ddp_series_theory.html) on Slurm-managed HPC is provided. Update the [shell script](scripts/slurm.sh) where required, make it executable (with `chmod +x scripts/slurm.sh`), and run it:
```
scripts/slurm.sh
```
This script will generate and submit a slurm job using `sbatch`. Generating the script dynamically allows resource requests to be set once at the top of the file, then passed to both slurm (to allocate resources) and Lightning (to utilise them).
