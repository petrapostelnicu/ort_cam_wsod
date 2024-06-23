#!/bin/sh

#SBATCH --job-name=train_vgg16
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load python
module load miniconda3
module load cuda/11.7

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate /scratch/ppostelnicu/rp/code/new-env-gpu
srun python main.py train_ort --classifier_name vgg16 > train_vgg16.log
conda deactivate
