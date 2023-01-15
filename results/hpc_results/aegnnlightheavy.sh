#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBATCH --job-name=aegnn
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuin0034
#SBATCH --output=results/aegnnlightheavy.out
#SBATCH --error=results/aegnnlightheavy.err


module purge
module load miniconda/3
conda activate aric
which python
conda activate aric
which python

nvidia-smi
python aegnnlightheavy.py