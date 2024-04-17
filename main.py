import pandas as pd
import json

train_claim = pd.read_csv("data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv", sep='\t')
print(train_claim)

train_stance = pd.read_json("data/English_train.json", orient='records', lines=True)
print(train_stance)

def cleanup_string(string):
    string = string.encode('ascii', 'ignore').decode('ascii')
    return string

def cleanup_object(input):
    if isinstance(input, str):
        return cleanup_string(input)
    elif isinstance(input, list):
        return [cleanup_object(item) for item in input]
    else:
        return input

def cleanup_dataframe(df):
    for col in df.columns:
        df[col] = df[col].apply(cleanup_object)

cleanup_dataframe(train_claim)
cleanup_dataframe(train_stance)

train_stance.to_json("data/cleaned_train.json", orient='records', lines=True)