
import pandas as pd
import json

train_claim = pd.read_csv("data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv", sep='\t')
print(train_claim)

with open("data/English_train.json") as json_file:
    json_strs = json_file.readlines()

json_datas = [json.loads(json_str) for json_str in json_strs]

train_stance = pd.DataFrame.from_dict(json_datas)
print(train_stance)

#train_stance = pd.read_json('data/English_train.json')
#print(train_stance)


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
#with open("data/cleaned_train.json", "w") as f:
#    json.dump(json_datas, f)


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