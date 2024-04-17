import pandas as pd
import json
from preprocessing import *

train_claim = pd.read_csv("data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv", sep='\t')
print(train_claim)

train_stance = pd.read_json("data/English_train.json", orient='records', lines=True)
print(train_stance)

cleanup_dataframe(train_claim)
cleanup_dataframe(train_stance)

train_stance.to_json("data/cleaned_train.json", orient='records', lines=True)