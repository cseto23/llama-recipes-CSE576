#!/bin/bash

#SBATCH -N 1 # number of nodes
#SBATCH -c 1 # number of cores
#SBATCH -t 0-2:00:00 # time in d-hh:mm:ss
#SBATCH --gres gpu:a100:1

module load mamba

source activate NLP_ENV

python3 inference.py --model_name $1 --peft_model $2 --split "train" --train_idx_start $3 \
--train_idx_end $4 --output_file $5