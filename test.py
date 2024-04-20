import pandas as pd

train_claim = pd.read_csv("data/checkworthy/english_train.tsv", sep='\t')

print(train_claim["Text"])