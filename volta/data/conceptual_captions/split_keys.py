import json
from toolz import partition_all
from translate_cc import PATH_DATA, load_data

def save(i, keys):
    with open(f"data/keys-part-{i}.json", "w") as f:
        json.dump(keys, f)

model_type = "m2m-100-lg"
folder_input = model_type + "-seed-1337"
data = load_data("train", "en", folder_input)
keys = list(data.keys())

for i, keys_ in enumerate(partition_all(400_000, keys)):
    save(i, keys_)
