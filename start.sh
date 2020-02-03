#!/bin/bash
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --threads-per-core=1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH -o job.%N.%j.out  # STDOUT
#SBATCH -e job.%N.%j.err  # STDERR
#SBATCH --job-name ASI_NIELS
#SBATCH --mem-per-cpu=20000

module purge
module load userspace/all userspace/custom opt/all
module load CUDA/10.0
module load CUDA/DNN10.1
module load anaconda3/2019.10

python data_ts.py
