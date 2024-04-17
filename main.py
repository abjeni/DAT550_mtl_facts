import pandas as pd
import json

train_claim = pd.read_csv("data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv", sep='\t')
print(train_claim)

train_stance = pd.read_json("data/English_train.json", orient='records', lines=True)
print(train_stance)

def deEmojify(input):
    if isinstance(input, str):
        return input.encode('ascii', 'ignore').decode('ascii')
    elif isinstance(input, list):
        return [deEmojify(item) for item in input]
    else:
        return input

for col in train_stance.columns:
    if train_stance[col].dtype == object:
        train_stance[col] = train_stance[col].apply(deEmojify)

train_stance.to_json("data/cleaned_train.json", orient='records', lines=True)