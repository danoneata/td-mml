import json
import pdb

from matplotlib import pyplot as plt

import click
import streamlit as st
import numpy as np

from toolz import assoc, concat


def load_data_dummy(*args, **kwargs):
    import random

    random.seed(1337)
    data = [
        {
            "result": random.choice(range(2)),
            "score_avg_mins": random.random(),
        }
        for _ in range(1000)
    ]


def load_data(model, lang):
    def func(key, values):
        assert len(values) == 1
        value = values[0]
        return assoc(value, "key", key)

    path = f"data/TTMML_eval_test_scores_marvl_{model}.json"
    with open(path, "r") as f:
        data = json.load(f)
        data = data[lang]
        data = [func(k, vs) for k, vs in data.items()]
        return data


@click.command()
@click.option("--lang", help="language")
def main(lang):
    models = [
        "translate_train_random_lg_filter",
        "translate_train_random_lg_full",
        "translate_train_vtlm_filter",
        "translate_train_vtlm_full",
        "zero_shot_random_lg",
        "zero_shot_vtlm",
    ]
    model = st.selectbox("model", models)
    data = load_data(model, lang)

    results = np.array([datum["result"] for datum in data])
    scores = np.array([float(datum["score_avg_mins"]) for datum in data])
    scores = np.log(scores + 0.5)

    num_bins = 20
    _, bins = np.histogram(scores, bins=num_bins)
    idxs = np.digitize(scores, bins)

    def get_frac_correct(i):
        r = results[idxs == i]
        if len(r) == 0:
            return 0
        else:
            return 100 * r.sum() / len(r)

    frac_correct = [get_frac_correct(i) for i in range(1, num_bins + 1)]

    # bins
    # frac_correct

    fig, ax = plt.subplots()
    width = 0.8 * (bins[1] - bins[0])
    ax.bar(bins[:-1], frac_correct, width=width)
    ax.set_xlabel("similarity to train data (log)")
    ax.set_ylabel("fraction correct (%)")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.hist(scores, bins=bins, align="left", width=width)
    ax.set_xlabel("similarity to train data (log)")
    ax.set_ylabel("counts")
    st.pyplot(fig)

    st.code(len(results))
    st.code(np.mean(results))

    idxs, = np.where(scores > 9)
    for i in idxs[:16]:
    # for i in range(16):
        data[i]


if __name__ == "__main__":
    main()
