import pandas as pd
import json
from preprocessing import *

train_claim = pd.read_csv("data/checkworthy/english_train.tsv", sep='\t')
print(train_claim)

train_stance = pd.read_json("data/stance/English_train.json", orient='records', lines=True)
print(train_stance)

cleanup_dataframe(train_claim)
cleanup_dataframe(train_stance)

train_stance.to_json("data/stance/cleaned_train.json", orient='records', lines=True)