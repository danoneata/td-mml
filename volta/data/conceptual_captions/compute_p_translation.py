import json
import pdb
import sys
import time

from tqdm import tqdm
from toolz import partition_all

import torch

from torch import nn

from transformers import AutoTokenizer

from translate_cc import load_model


def load_data(lang):
    with open(f"data/cc/analysis-{lang}-cache.json") as f:
        return json.load(f)


def save_data(data, lang):
    with open(f"data/cc/analysis-cache-2022-05-12/{lang}.json", "w") as f:
        json.dump(data, f, indent=4)


model_type = "m2m-100-lg"
device = "cuda:5"

lang = sys.argv[1]
assert lang in "id sw ta tr zh".split()

data = load_data(lang)

model = load_model(model_type, device)
model.translator.model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/m2m100_418M", src_lang="en", tgt_lang=lang
)
kwargs = dict(return_tensors="pt", padding=True, truncation=True)


cross_entropy = nn.CrossEntropyLoss()


def compute_loss(data):
    # prepare data
    text_src = [datum["text-src"] for datum in data]
    text_tgt = [datum["text-tgt"] for datum in data]

    model_inputs = tokenizer(text_src, **kwargs)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_tgt, **kwargs).input_ids

    model_inputs.to(device)
    labels = labels.to(device)

    # forward pass
    with torch.no_grad():
        output = model.translator.model(**model_inputs, labels=labels)

    logits = output["logits"].to("cpu")
    labels = labels.to("cpu")

    data_out = data.copy()
    EOS_TOKEN = 2

    for i in range(len(data)):
        n = labels[i].tolist().index(EOS_TOKEN) + 1
        data_out[i]["m2m-100-lg"] = {
            "loss": cross_entropy(logits[i][:n], labels[i][:n]).item(),
            "num-tokens-tgt": n,
        }

    return data_out


data_out = []
batch_size = 32
start_time = time.time()

for batch in tqdm(list(partition_all(batch_size, data))):
    batch_out = compute_loss(list(batch))
    data_out.extend(batch_out)

print(lang)
print("time elapsed: {:.3f}".format(time.time() - start_time))
print()

save_data(data_out, lang)
