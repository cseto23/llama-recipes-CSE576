#!/bin/bash

#SBATCH -N 1 # number of nodes
#SBATCH -c 1 # number of cores
#SBATCH -t 0-4:00:00 # time in d-hh:mm:ss
#SBATCH --gres gpu:a100:1
#SBATCH --mem=80GB

module load mamba

source activate NLP_ENV

python3 comparing.py --model_path /scratch/mrlunt/models--lmsys--vicuna-33b-v1.3/snapshots/ef8d6becf883fb3ce52e3706885f761819477ab4/ --num_worse 500 --combined_outputs_filepath ~/CSE_576_2023F_project_1/diverse_train_model/predictions/alpaca/combined_outputs.json --save_filepath ~/CSE_576_2023F_project_1/diverse_train_model/predictions/alpaca/output_results.csv

