import json
import pdb
import random

random.seed(1337)

with open("data/cc/annotations/caption_valid.json", "r") as f:
    data = json.load(f)

keys = random.sample(list(data.keys()), 1000)

with open("data/validation-1000.txt", "w") as f:
    for key in keys:
        f.write(key + "\n")
