"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
import nlp

configuration = transformers.BertConfig()
model = transformers.BertModel(configuration)
configuration = model.config

train_claim = pd.read_csv("data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv", sep='\t')

dataset_dict = {
    "claim": nlp.Dataset.from_pandas(train_claim)
}

print(dataset_dict["claim"])
"""

import torch
from transformers import AutoTokenizer, AutoModel

device = "cpu" # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

model = AutoModel.from_pretrained(
    "distilbert/distilbert-base-uncased",
    torch_dtype=torch.float,
#    attn_implementation="flash_attention_2"
)

text = "deez nutz"

encoded_input = tokenizer(text, return_tensors='pt').to(device)

model.to(device)

output = model(**encoded_input)

print(encoded_input)

print(output)