import json
import os
import pdb
import random

import streamlit as st
import seaborn as sns

from matplotlib import pyplot as plt


random.seed(1337)
st.set_page_config(layout="wide")


def agg_scores(scores, weights):
    return sum(weights[k] * scores[k] for k in scores.keys())


langs = "id sw ta tr zh".split()  # MaRVL languages
weights = {
    "len-ratio": 1.0,
    "sim-tgt-src": 1.0,
    "uniformity": 1.0,
}
sort_funcs = {
    "aggregated-score": agg_scores,
    "len-ratio": lambda s, *_: s["len-ratio"],
    "sim-tgt-src": lambda s, *_: s["sim-tgt-src"],
    "uniformity": lambda s, *_: s["uniformity"],
}

with st.sidebar:
    lang = st.selectbox("language", langs, index=0)
    st.markdown("---")

    help_text = {
        "len-ratio": "number of characters in the target language divided by number of characters in the source language",
        "sim-tgt-src": "cosine similarity between the bag of token representation of the source and target texts",
        "uniformity": "number of unique tokens in the translation divided by total number of tokens",
    }

    st.markdown("""
    - score each translation based on three features (see help)
    - adjust their weights using the sliders below
    """)
    for feature in weights.keys():
        weights[feature] = st.number_input(feature, value=1.0, help=help_text[feature])
    st.markdown("---")

    sort_by = st.selectbox("sort by", sort_funcs.keys(), index=0)


def load_scored_data(lang):

    cache_path = f"data/cc/analysis-{lang}-cache.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    from collections import Counter

    import numpy as np

    from scipy.spatial.distance import cosine
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from translate_cc_full import load_data, save_data

    split = "train"
    folder = "m2m-100-lg-full"

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    ivocab = {i: k for k, i in tokenizer.vocab.items()}

    data_src = load_data(split, "en", folder)
    data_tgt = load_data(split, lang, folder)

    # keys = list(data_src.keys())

    # subsample data
    keys = list(random.sample(data_src.keys(), 2048))
    # save_data({k: data_src[k] for k in keys}, split, "en", "m2m-100-lg-tiny")
    # save_data({k: data_tgt[k] for k in keys}, split, lang, "m2m-100-lg-tiny")
    # pdb.set_trace()

    tokens_src = {key: tokenizer(data_src[key])["input_ids"] for key in keys}
    tokens_tgt = {key: tokenizer(data_tgt[key])["input_ids"] for key in keys}

    K = 250_002

    def get_uniformity(key):
        tokens = tokens_tgt[key] 
        counts = Counter(tokens)  # unique tokens
        return 1 - len(counts) / len(tokens)

    def get_similarity_tgt_src(key):
        def bow(idxs):
            v = np.zeros(K)
            for i in idxs:
                # ignore punctuation
                if i in (0, 2, 4, 5, 6, 9, 12, 15, 16, 20, 25, 26, 27, 30, 32, 38, 63, 64, 74, 94, 1104):
                    continue
                v[i] += 1
            return v

        tgt = bow(tokens_tgt[key])
        src = bow(tokens_src[key])

        return 1 - cosine(tgt, src)

    def get_length_ratio(key):
        return len(data_tgt[key]) / len(data_src[key])

    def get_scores(key):
        return {
            "len-ratio": get_length_ratio(key),
            "sim-tgt-src": get_similarity_tgt_src(key),
            "uniformity": get_uniformity(key),
        }

    scored_keys = [
        {
            "key": key,
            "scores": get_scores(key),
            "text-src": data_src[key],
            "text-tgt": data_tgt[key],
        }
        for key in tqdm(keys)
    ]

    with open(cache_path, "w") as f:
        json.dump(scored_keys, f, indent=4)

    return scored_keys


data = load_scored_data(lang)
data = sorted(data, reverse=True, key=lambda d: sort_funcs[sort_by](d["scores"], weights))

scores = [agg_scores(d["scores"], weights) for d in data]
fig, ax = plt.subplots()
sns.ecdfplot(scores, ax=ax)
ax.set_xlabel("aggregated score")
ax.set_ylabel("proportion")

st.markdown("### Cumulative distribution of scores")
col, _ = st.columns([4, 4])
col.pyplot(fig)
st.markdown("---")

st.markdown("### Ranked samples")
st.markdown("- the translations sorted in decreasing of the weighted sum of the three features")
for rank, datum in enumerate(data, 1):
    score = agg_scores(datum["scores"], weights)
    str_scores = " · ".join("{}: {:.1f}".format(k, v) for k, v in datum["scores"].items())
    st.markdown("{} ◇ `{}` ◇ score: {:.3f} ← ".format(rank, datum["key"], score) + str_scores)
    st.code("en: {}\n{}: {}".format(datum["text-src"], lang, datum["text-tgt"]))
    st.markdown("---")
