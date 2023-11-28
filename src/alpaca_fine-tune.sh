#!/bin/bash

#SBATCH -p general
#SBATCH -N 1 # number of nodes
#SBATCH -c 1 # number of cores
#SBATCH -t 4-12:00:00 # time in d-hh:mm:ss
#SBATCH --gres gpu:a100:1

module load mamba

source activate NLP_ENV

python3 -m llama_recipes.finetuning --use_peft --save_model --peft_method lora --quantization --num_epochs 5 --dataset alpaca_dataset --model_name $1 --output_dir $2

<<Block_comment
torchrun --nnodes 1 --nproc_per_node 2  examples/finetuning.py --enable_fsdp --use_peft --save_model --peft_method lora --quantization --dataset alpaca_dataset --model_name $1 --output_dir $2
Block_comment