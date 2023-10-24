import json
from datasets import load_dataset

def get_custom_dataset(dataset_config, tokenizer, split: str):
    fp = open('diverse_examples.json','r')
    div_ex = json.load(fp)
    fp.close()

    div_ex = div_ex['examples']

    alpaca_ds = load_dataset('tatsu-lab/alpaca',split=split)
    diverse_ds = []
    for i in div_ex:
        diverse_ds.append(alpaca_ds[i]['text'])
    
    return diverse_ds
