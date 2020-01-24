#!/bin/bash
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --threads-per-core=1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH -o job.%N.%j.out  # STDOUT
#SBATCH -e job.%N.%j.err  # STDERR
#SBATCH --job-name ASI_NIELS

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
