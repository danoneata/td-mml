import json
import pdb

from matplotlib import pyplot as plt

import click
import streamlit as st
import numpy as np
import seaborn as sns

from toolz import assoc, concat


sns.set_theme()


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
    def func(datum):
        # for some reason, a result of 0 means that the prediction is correct
        return assoc(datum, "result", 1 - datum["result"])

    path = f"data/TTMML_eval_test_scores_marvl_{model}.json"
    with open(path, "r") as f:
        data = json.load(f)
        data = data[lang]
        data = [func(datum) for datum in data]
        return data


def plot1(data, num_bins=10, use_log=False):
    results = np.array([datum["result"] for datum in data])
    scores = np.array([float(datum["score_avg"]) for datum in data])

    if use_log:
        scores = np.log(scores + 0.5)
        xlabel = "(log)"
    else:
        scores = 10 * scores / scores.max()
        xlabel = ""

    _, bins = np.histogram(scores, bins=num_bins)
    idxs = np.digitize(scores, bins)

    def get_frac_correct(i):
        r = results[idxs == i]
        if len(r) == 0:
            return 0
        else:
            return r.sum() / len(r)

    frac_correct = [get_frac_correct(i) for i in range(1, num_bins + 1)]
    frac_correct = np.array(frac_correct)

    # bins
    # frac_correct

    counts = [np.sum(idxs == i) for i in range(1, 1 + num_bins)]
    yerr = np.sqrt(frac_correct * (1 - frac_correct) / counts)

    fig, ax = plt.subplots()
    width = 0.8 * (bins[1] - bins[0])
    ax.bar(bins[:-1], frac_correct, width=width, yerr=yerr)
    ax.set_xlabel("similarity to train data {}".format(xlabel))
    ax.set_ylabel("accuracy")
    sns.kdeplot(x=scores, ax=ax, color="orange", fill=True, alpha=0.3)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.hist(scores, bins=bins, align="left", width=width)
    ax.set_xlabel("similarity to train data {}".format(xlabel))
    ax.set_ylabel("counts")
    st.pyplot(fig)

    st.code(len(results))
    st.code(np.mean(results))

    idxs, = np.where(scores > 9)
    # for i in idxs[:16]:
    for i in range(16):
        data[i]

def plotq(data, num_bins=10, use_log=False, title=""):
    results = np.array([datum["result"] for datum in data])
    scores = np.array([float(datum["score_avg"]) for datum in data])

    if use_log:
        scores = np.log(scores + 0.5)
        xlabel = "(log)"
    else:
        scores = 10 * scores / scores.max()
        xlabel = ""

    Δ = 1.0 / num_bins
    bins = [np.quantile(scores, q) for q in np.arange(0.0, 1.0 + Δ, Δ)]
    idxs = np.digitize(scores, bins)

    def get_frac_correct(i):
        r = results[idxs == i]
        if len(r) == 0:
            return 0
        else:
            return r.sum() / len(r)

    frac_correct = [get_frac_correct(i) for i in range(1, num_bins + 1)]
    frac_correct = np.array(frac_correct)

    # bins
    # frac_correct

    counts = [np.sum(idxs == i) for i in range(1, 1 + num_bins)]
    yerr = np.sqrt(frac_correct * (1 - frac_correct) / counts)

    fig, ax = plt.subplots()

    width = 0.8
    qbins = np.arange(len(bins) - 1)

    ax.bar(qbins, frac_correct, width=width)
    ax.set_xlabel("similarity to train data {}".format(xlabel))
    ax.set_ylabel("accuracy")
    ax.set_xticks(qbins)
    ax.set_xticklabels([f"$Q_{int(i):d}$" for i in range(1, num_bins + 1)])
    ax.set_title(title)
    # sns.kdeplot(x=scores, ax=ax, color="orange", fill=True, alpha=0.3)
    st.pyplot(fig)


def main():
    langs = "id sw ta tr zh".split()
    models = [
        "translate_train_random_lg_filter",
        "translate_train_random_lg_full",
        "translate_train_vtlm_filter",
        "translate_train_vtlm_full",
        "zero_shot_random_lg",
        "zero_shot_vtlm",
    ]

    with st.sidebar:
        lang = st.selectbox("language", langs)
        model = st.selectbox("model", models)

    data = load_data(model, lang)
    plotq(data, num_bins=5, title=lang)
    plot1(data)


if __name__ == "__main__":
    main()
