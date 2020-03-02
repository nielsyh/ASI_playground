#!/bin/bash
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --threads-per-core=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH -o job.%N.%j.out  # STDOUT
#SBATCH -e job.%N.%j.err  # STDERR


echo what model: rf, ann, lstm, flow
read -p 'model: ' model
echo set $model

echo Single 0 or Multi 1 output model?
read -p 'Multi or Single : ' output
echo set $output

module purge
module load userspace/all userspace/custom opt/all
module load anaconda3/2019.10
module load CUDA/10.1
module load CUDA/DNN10.1


if [[ $model = "rf" ]]
then
   if [[ $output = "0" ]]
    then
        echo single rf
        #SBATCH -J RF s
        python experiments_single_rf.py
    else
        echo multi rf
        #SBATCH -J RF m
        python experiments_multi_rf.py
    fi
fi

if [[ $model = "ann" ]]
then
   if [[ $output = "0" ]]
    then
        echo single ann
        #SBATCH -J ANN s
        python experiments_single_ann.py
    else
        echo multi ann
        #SBATCH -J ANN m
        python experiments_multi_ann.py
    fi
fi

if [[ $model = "lstm" ]]
then
   if [[ $output = "0" ]]
    then
        echo single lstm
        #SBATCH -J LSTMs
        python experiments_single_lstm.py
    else
        echo multi
        #SBATCH -J LSTMm
        python experiments_multi_lstm.py
    fi
fi

if [[ $model = "flow" ]]
then
    echo flow
    #SBATCH -J flow
    python experiments_flow.py
fi