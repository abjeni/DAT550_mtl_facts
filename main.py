import pandas as pd
import json
from preprocessing import *

train_claim = pd.read_csv("data/checkworthy/english_train.tsv", sep='\t')
print(train_claim)

train_stance = pd.read_json("data/stance/english_train.json", orient='records', lines=True)
print(train_stance)

dev_stance = pd.read_json("data/stance/english_dev.json", orient='records', lines=True)
print(train_stance)

cleanup_dataframe(dev_stance)
cleanup_dataframe(train_stance)

train_stance.to_json("data/stance/cleaned_train.json", orient='records', lines=True)
dev_stance.to_json("data/stance/cleaned_dev.json", orient='records', lines=True)

def create_new_stance(df):
    
    new_stance = pd.DataFrame(columns=["rumor", "label"])

    for index, row in df.iterrows():
        if row["label"] != "NOT ENOUGH INFO":
            new_stance.loc[len(new_stance.index)] = row[["rumor", "label"]]

            for evidence in row["evidence"]:
                new_stance.loc[len(new_stance.index)] = [evidence[2], "SUPPORTS"]

            #print(row["evidence"])
    
    return new_stance

new_dev_stance = create_new_stance(dev_stance)
new_train_stance = create_new_stance(train_stance)

new_dev_stance.to_csv("data/stance/cleaned_dev.tsv", sep='\t')
new_train_stance.to_csv("data/stance/cleaned_train.tsv", sep='\t')