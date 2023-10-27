# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
import os

currDir = os.getcwd()
currDirList = currDir.split('/')

if currDirList[len(currDirList)-1] == 'src':
    currDirList.pop()
    currDir = '/'.join(currDirList)

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = currDir+"/src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = currDir+"/src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = currDir+"/src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = currDir+"/examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class diverse_dataset:
    dataset: str = "diverse_dataset"
    file: str = currDir+"/examples/diverse_dataset.py"
    train_split: str = "train"
    test_split: str = "val"