
import pandas

english_train_checkworthy = pandas.read_csv("data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv", sep='\t')
print(english_train_checkworthy)


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