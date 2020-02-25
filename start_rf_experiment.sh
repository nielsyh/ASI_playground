#!/bin/bash
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --threads-per-core=1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH -o job.%N.%j.out  # STDOUT
#SBATCH -e job.%N.%j.err  # STDERR
#SBATCH -J RF

module purge
module load userspace/all userspace/custom opt/all
module load anaconda3/2019.10
module load CUDA/10.1
module load CUDA/DNN10.1

python experiments_sequence_RF.py $1 $2 $3


