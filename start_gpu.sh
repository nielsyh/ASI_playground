#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -o job.%N.%j.out  # STDOUT
#SBATCH -e job.%N.%j.err  # STDERR
#SBATCH --job-name ASI_NIELS_GPU

module purge
module load userspace/all userspace/custom opt/all
module load gcc/7.2.0
module load openmpi/gcc72/mlnx/3.0.0
module load CUDA/10.0
module load CUDA/DNN10.1
#module load anaconda3/2019.03

conda init --all
conda activate my_env
python3.6 main.py
