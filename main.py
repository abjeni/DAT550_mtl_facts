
import pandas as pd
import json

train_checkworthy = pd.read_csv("data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv", sep='\t')
print(train_checkworthy)

with open("data/English_train.json") as json_file:
    json_strs = json_file.readlines()

json_datas = [json.loads(json_str) for json_str in json_strs]

train_stance = pd.DataFrame.from_dict(json_datas)
print(train_stance)

#train_stance = pd.read_json('data/English_train.json')
#print(train_stance)

"""
with open("data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv") as train_file:
    lines = train_file.readlines()

df_t = []
i = 0
for line in lines:
    #df_t.append(line.split("\t"))
    if len(line.split("\t")) != 3:
        print(f"no {i}")
    i += 1

print(len(lines[1].split("\t")))
"""