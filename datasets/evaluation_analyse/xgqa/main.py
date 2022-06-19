import json
import pdb

import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd

from matplotlib import pyplot as plt
from itertools import product
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from toolz import assoc, merge
from tqdm import tqdm


sns.set_theme()


languages = "bn de en id ko pt ru zh".split()


def load_predictions(lang):
    # path = f"data/results/vtlm_20langs_xgqa_test_results/xGQA{lang}-random_lg-batch_size_256-step_100000-langs_20/pytorch_model_best.bin-/test_{lang}_result.json"
    path = f"data/results-new/vtlm_20langs_xgqa_test_results/xGQA{lang}-vtlm-batch_size_256-step_100000-langs_20/pytorch_model_best.bin-/test_{lang}_result.json"
    # path = f"data/results-new/vtlm_20langs_xgqa_test_results/xGQA{lang}-vtlm-batch_size_256-step_100000-langs_20-val_epoch_0/pytorch_model_epoch_0_step_3683.bin-/test_{lang}_result.json"
    with open(path, "r") as f:
        return json.load(f)


def load_groundtruth(lang):
    path = f"../../xGQA/annotations/zero_shot/testdev_balanced_questions_{lang}.json"
    with open(path, "r") as f:
        return json.load(f)


def merge_true_pred(true, pred):
    keys = ["question", "answer", "imageId"]
    return [merge(p, {k: true[p["questionId"]][k] for k in keys}) for p in tqdm(pred)]


@st.cache
def load_data(lang):
    true = load_groundtruth(lang)
    pred = load_predictions(lang)
    return merge_true_pred(true, pred)


def show1(datum):
    datum


def demo():
    lang = st.selectbox("language", languages)
    data = load_data(lang)
    for i in range(16):
        show1(data[i])


def cohen_kappa_score_max(y1, y2):
    confusion = confusion_matrix(y1, y2)
    n_classes = confusion.shape[0]
    assert n_classes == 2

    p = confusion / confusion.sum()
    p_row = p.sum(axis=0)
    p_col = p.sum(axis=1)

    # p_max = p[0, 0] + p[1, 1]
    p_max = np.sum(np.minimum(p_row, p_col))
    p_exp = np.sum(p_row * p_col)
    return (p_max - p_exp) / (1 - p_exp)


def agreement_across_languages():
    def is_correct(datum):
        return datum["answer"] == datum["prediction"]

    data = [
        merge(datum, {"is-correct": is_correct(datum), "lang": lang})
        for lang in languages
        for datum in load_data(lang)
    ]

    df = pd.DataFrame(data)
    df
    # pdb.set_trace()

    results = df.pivot("questionId", "lang", "is-correct")
    "is the prediction correct for the given question in language x?"
    results

    "double checking that I get the same accuracy as Chen:"
    for lang in languages:
        lang, 100 * results[lang].mean()
        pass

    agreement = np.vstack(
        [
            [cohen_kappa_score(results[lang1], results[lang2]) for lang2 in languages]
            for lang1 in languages
        ]
    )
    agreement_max = np.vstack(
        [
            [cohen_kappa_score_max(results[lang1], results[lang2]) for lang2 in languages]
            for lang1 in languages
        ]
    )
    agreement = agreement / agreement_max

    # agreement

    mask = np.zeros_like(agreement)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(
            agreement,
            # mask=mask,
            square=True,
            annot=True,
            fmt=".2f",
            xticklabels=languages,
            yticklabels=languages,
            vmin=0.0,
            vmax=1.0,
        )
        
        "Cohen's Kappa score between pairs of langauages"
        st.pyplot(fig)

    df = df.pivot("questionId", "lang")

    n = 64
    st.markdown("---")
    st.markdown(f"below we show a random sample of {n} examples")
    df = df.sample(n)

    # st.markdown(f"samples for which en is wrong and the others correct")
    # idxs = (
    #     ~df["is-correct"]["en"] &
    #     df["is-correct"]["bn"] &
    #     df["is-correct"]["de"] &
    #     df["is-correct"]["id"] &
    #     df["is-correct"]["ko"] &
    #     df["is-correct"]["pt"] &
    #     df["is-correct"]["ru"] &
    #     df["is-correct"]["zh"])
    # df = df[idxs]

    for index, row in df.iterrows():
        st.markdown("### question id: `{}`".format(index))
        st.image("../../xGQA/images/{}.jpg".format(row["imageId"][0]))
        # for lang in languages:
        #     st.markdown(lang)
        #     st.markdown(row["question"][lang])
        #     st.markdown(row["prediction"][lang])
        #     st.markdown(row["is-correct"][lang])
        st.code("""Q: {}
A: {}""".format(row["question"]["en"], row["answer"]["en"]))
        rr = row[["is-correct", "prediction", "question"]].unstack().T
        rr
        st.markdown("---")
    pdb.set_trace()


def main():
    agreement_across_languages()


if __name__ == "__main__":
    main()
