#!/bin/bash

#SBATCH -p general
#SBATCH -N 1 # number of nodes
#SBATCH -c 1 # number of cores
#SBATCH -t 0-3:00:00 # time in d-hh:mm:ss
#SBATCH --gres gpu:a100:1

module load mamba

source activate NLP_ENV

python3 -m llama_recipes.finetuning --dataset "diverse_dataset" --custom_dataset.file $1 --use_peft --save_model --peft_method lora --quantization --num_epochs 5 --model_name $2 --output_dir $3