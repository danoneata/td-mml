# See the following notebook for more details
# https://colab.research.google.com/drive/1bH3vyF6YhniM7XVXyIHoiDpHN57Kth1O?usp=sharing
import json
import os
import pdb
import random

from collections import Counter
from matplotlib import pyplot as plt

import numpy as np
import streamlit as st

import torch
from torch.optim import Adam
from torch.nn import functional as F


random.seed(1)
device = "cpu"


def load_data():
    def load_langs():
        with open("data/langs-common.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    def load_data_lang(lang):
        print("Loading data for", lang)
        with open(f"data/cc/translations/m2m-100-md-seed-1337/{lang}-train.json", "r") as f:
            return json.load(f)
    cache_path = "data/cc/translations/lang-and-sample.npz"
    if os.path.exists(cache_path):
        data_all = np.load(cache_path)
        data = data_all["data"]
        langs = data_all["langs"]
        samples = data_all["samples"]
        return data.astype(np.float64), langs, samples
    langs = load_langs()
    samples = list(load_data_lang("en").keys())
    data = []
    for lang in langs:
        data_lang = load_data_lang(lang)
        data.append([1 if sample in data_lang else 0 for sample in samples])
    data = np.vstack(data)
    # for storing purposes use a more efficient data type
    np.savez(cache_path, data=data.astype(np.uint8), langs=langs, samples=samples)
    return data.astype(np.float64), langs, samples


def load_data_synthetic():
    LANG_PROPS = [1, 0.8, 0.75, 0.5, 0.3, 0.2, 0.1]
    NUM_SENTS = 5000
    NUM_LANGS = len(LANG_PROPS)
    langs = "en ja id ro sw uk ta".split()
    samples = [str(i) for i in range(5000)]
    data = np.zeros((NUM_LANGS, NUM_SENTS))
    for l, p in enumerate(LANG_PROPS):
        for s in range(NUM_SENTS):
            data[l, s] = random.random() <= p
    return data, langs, samples


class ProbasModel(torch.nn.Module):
    def __init__(self, data):
        super(ProbasModel, self).__init__()
        self.data = torch.tensor(data).to(device)
        self.num_langs, self.num_samples = data.shape
        self.params = torch.nn.Parameter(torch.ones(self.num_langs, self.num_samples))

    def get_probas(self):
        p = F.relu(self.params) * self.data
        return p / p.sum()

    def forward(self, q_l):
        p_s = self.get_probas().sum(0)
        p_l = self.get_probas().sum(1)
        # WARN `kl_div` requires the first argument in the log space and the
        # second one in the probabilitys space.
        unif_s = torch.ones(self.num_samples).double() / self.num_samples
        unif_s = unif_s.to(device)
        return (
            F.kl_div(p_l.log(), q_l, reduction="batchmean") +
            F.kl_div(p_s.log(), unif_s, reduction="batchmean")
        )


def compute_p_lang_given_sample_adjusted():
    data, langs, samples = load_data()

    p_lang_and_sample = data / data.sum()
    p_lang = data.sum(axis=1)

    # the adjusted probability distribution over the languages will be the
    # target probability
    α = 0.3
    q_lang = np.array([p ** α for p in p_lang])
    q_lang = q_lang / q_lang.sum()

    probas_model = ProbasModel(data).to(device)
    q_lang = torch.tensor(q_lang).double().to(device)

    optimizer = Adam(probas_model.parameters(), lr=0.06)
    num_steps = 5000
    print_every = 1

    for i in range(num_steps):
        loss = probas_model.forward(q_lang)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_every == 0:
            print(f"{i:4d}", loss.item())

    p_lang_and_sample_new = probas_model.get_probas().detach().cpu().numpy().astype(np.float32)

    # store the joint probability instead of the conditional, since the former
    # is more informative
    path = f"data/cc/translations/p-lang-and-sent-alpha-{α}.npz"
    np.savez(path, p_lang_and_sample=p_lang_and_sample_new, langs=langs, sents=samples)

    # data_json = {
    #     sample: {
    #         lang: p_lang_given_sample_new[l, s]
    #         for l, lang in enumerate(langs)
    #     }
    #     for s, sample in enumerate(samples)
    # }
    # with open(f"data/cc/translations/p-lang-given-sample-alpha-{α}.json", "w") as f:
    #     json.dump(data_json, f)

    # p_sample = p_lang_and_sample.sum(axis=0)
    # p_lang_given_sample = p_lang_and_sample / p_sample[np.newaxis]

    # p_sample_new = p_lang_and_sample_new.sum(axis=0)
    # p_lang_given_sample_new = p_lang_and_sample_new / p_sample_new[np.newaxis]

    # fig, axs = plt.subplots(nrows=2)
    # axs[0].imshow(p_lang_given_sample[:, :50])
    # axs[1].imshow(p_lang_given_sample_new[:, :50])
    # st.pyplot(fig)


def main():
    compute_p_lang_given_sample_adjusted()


if __name__ == "__main__":
    main()
