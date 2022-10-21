import json
import pdb

import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd

from matplotlib import pyplot as plt
from itertools import groupby, product
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from toolz import assoc, first, merge
from tqdm import tqdm


sns.set_theme()


languages = "bn de en id ko pt ru zh".split()

MODEL_TO_PATH = {
    "tt-mml-pretrain-no-vtlm": "data/results/vtlm_20langs_xgqa_test_results/xGQA{lang}-random_lg-batch_size_256-step_100000-langs_20/pytorch_model_best.bin-/test_{lang}_result.json",
    "tt-mml-pretrain": "data/results-new/vtlm_20langs_xgqa_test_results/xGQA{lang}-vtlm-batch_size_256-step_100000-langs_20/pytorch_model_best.bin-/test_{lang}_result.json",
    "tt-mml-pretrain-epoch-1": "data/results-new/vtlm_20langs_xgqa_test_results/xGQA{lang}-vtlm-batch_size_256-step_100000-langs_20-val_epoch_0/pytorch_model_epoch_0_step_3683.bin-/test_{lang}_result.json",
    "tt-mml-pretrain-finetune": "data/results/preds_xGQA_multilingual_eval/ctrl_ttmml_vtlm_20langs/test/results/ctrl_xuniter_base/xGQA{lang}-vtlm-batch_size_256-step_100000-langs_20-eval_step_29468-filter/pytorch_model_epoch_0_step_29468.bin-/test_{lang}_result.json",
    #
    "tt-mml-pretrain-epoch-22-5": "data/zero_shot/ctrl_ttmml_vtlm_20langs/xgqa/results/xGQA{lang}-vtlm-batch_size_256-step_220000-langs_20/pytorch_model_best.bin-/test_{lang}_result.json",
    "tt-mml-pretrain-epoch-22": "data/results_epoch22/zero_shot/ctrl_ttmml_vtlm_20langs/xgqa/results/xGQA{lang}-vtlm-batch_size_256-step_220000-langs_20-val_epoch_0/pytorch_model_epoch_0_step_3683.bin-/test_{lang}_result.json",
    "tt-mml-pretrain-epoch-22-finetune": "data/results_epoch22/translate_train/ctrl_ttmml_vtlm_20langs/xgqa/results/ctrl_xuniter_base/xGQA{lang}-vtlm-batch_size_256-step_220000-langs_20-eval_step_29468-filter/pytorch_model_epoch_0_step_29468.bin-/test_{lang}_result.json",
}


def load_predictions(model, lang):
    path = MODEL_TO_PATH[model].format(lang=lang)
    with open(path, "r") as f:
        return json.load(f)


def load_groundtruth(lang):
    path = f"data/annotations/zero_shot/testdev_balanced_questions_{lang}.json"
    with open(path, "r") as f:
        return json.load(f)


def merge_true_pred(true, pred):
    keys = ["question", "answer", "imageId"]
    return [merge(p, {k: true[p["questionId"]][k] for k in keys}) for p in tqdm(pred)]


# @st.cache(allow_output_mutation=True)
def load_data(model, lang):
    true = load_groundtruth(lang)
    pred = load_predictions(model, lang)
    return merge_true_pred(true, pred)


def show1(datum):
    datum


def demo():
    lang = st.selectbox("language", languages)
    data = load_data("tt-mml-pretrain", lang)
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


def cohen_kappa_score_within(y1, y2):
    return cohen_kappa_score(y1, y2) / cohen_kappa_score_max(y1, y2)


def is_correct(datum):
    return datum["answer"] == datum["prediction"]


def agreement_across_languages():
    data = [
        merge(datum, {"is-correct": is_correct(datum), "lang": lang})
        for lang in languages
        for datum in load_data("tt-mml-pretrain-epoch-22-5", lang)
        # for datum in load_data("tt-mml-pretrain-epoch-22-finetune", lang)
        # for datum in load_data("tt-mml-pretrain", lang)
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
            [
                cohen_kappa_score_within(results[lang1], results[lang2])
                for lang2 in languages
            ]
            for lang1 in languages
        ]
    )
    agreement

    mask = np.zeros_like(agreement)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            agreement,
            square=True,
            annot=True,
            fmt=".2f",
            xticklabels=languages,
            yticklabels=languages,
            vmin=0.0,
            vmax=1.0,
            ax=ax,
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
        st.code(
            """Q: {}
A: {}""".format(
                row["question"]["en"], row["answer"]["en"]
            )
        )
        rr = row[["is-correct", "prediction", "question"]].unstack().T
        rr
        st.markdown("---")
    pdb.set_trace()


def agreement_delta():
    def compute_agreement(model):
        data = [
            merge(datum, {"is-correct": is_correct(datum), "lang": lang})
            for lang in languages
            for datum in load_data(model, lang)
        ]

        df = pd.DataFrame(data)
        results = df.pivot("questionId", "lang", "is-correct")

        return np.vstack(
            [
                [
                    cohen_kappa_score_within(results[lang1], results[lang2])
                    for lang2 in languages
                ]
                for lang1 in languages
            ]
        )

    agreement_1 = compute_agreement("tt-mml-pretrain")
    agreement_2 = compute_agreement("tt-mml-pretrain-finetune")
    agreement_Δ = agreement_2 - agreement_1

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            agreement_1,
            square=True,
            annot=True,
            fmt=".2f",
            xticklabels=languages,
            yticklabels=languages,
            vmin=0.0,
            vmax=1.0,
            # ax=axs[0],
            cbar=False,
        )
        ax.set_title("finetune: english-only")
        plt.savefig("output/xgqa/cross-language-correlation-tt-mml-en.png")
        st.pyplot(fig)

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            agreement_2,
            square=True,
            annot=True,
            fmt=".2f",
            xticklabels=languages,
            yticklabels=languages,
            vmin=0.0,
            vmax=1.0,
            # ax=axs[1],
            cbar=False,
        )
        ax.set_title("finetune: translations")
        plt.savefig("output/xgqa/cross-language-correlation-tt-mml-tt.png")
        st.pyplot(fig)

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            agreement_Δ,
            square=True,
            annot=True,
            fmt=".2f",
            xticklabels=languages,
            yticklabels=languages,
            vmin=0.0,
            vmax=1.0,
            # ax=axs[2],
            cbar=False,
        )
        ax.set_title("difference")
        plt.savefig("output/xgqa/cross-language-correlation-diff.png")
        st.pyplot(fig)


def cross_model_correlations():
    data1 = [
        merge(datum, {"is-correct": is_correct(datum), "lang": lang})
        for lang in languages
        for datum in load_data("tt-mml-pretrain", lang)
    ]
    data2 = [
        merge(datum, {"is-correct": is_correct(datum), "lang": lang})
        for lang in languages
        for datum in load_data("tt-mml-pretrain-finetune", lang)
    ]

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    results1 = df1.pivot("questionId", "lang", "is-correct")
    results2 = df2.pivot("questionId", "lang", "is-correct")

    y = np.array(
        [cohen_kappa_score_within(results1[lang], results2[lang]) for lang in languages]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(y=y, x=languages, ax=ax)
    st.pyplot(fig)
    plt.savefig("output/xgqa/cross-model-correlation-barplot.png")

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            np.diag(y),
            square=True,
            annot=True,
            fmt=".2f",
            xticklabels=languages,
            yticklabels=languages,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            mask=1 - np.eye(len(y)),
        )
        st.pyplot(fig)
        plt.savefig("output/xgqa/cross-model-correlation-heatmap.png")


def ensemble_results(model):
    data = [
        merge(datum, {"is-correct": is_correct(datum), "lang": lang})
        for lang in languages
        for datum in load_data(model, lang)
    ]

    def most_common(xs):
        return max(set(xs), key=xs.count)

    data = sorted(data, key=lambda datum: datum["questionId"])
    results = []
    for k, group in groupby(data, lambda datum: datum["questionId"]):
        group = list(group)
        preds = [datum["prediction"] for datum in group]
        results.append(most_common(preds) == group[0]["answer"])

    st.markdown("Ensemble results")
    st.markdown("{}: {:.1%}".format(model, np.mean(results)))


def random_agreement():
    # agreement_delta()
    # cross_model_correlations()

    import random

    n = 1000
    p = 0.1

    y1 = [random.random() > p for _ in range(n)]
    y2 = [random.random() > p for _ in range(n)]

    st.code(np.sum(y1))
    st.code(cohen_kappa_score_within(y1, y2))


def show_qualitative_agreement_increase():
    def agreement(results):
        predictions = [result["prediction"] for result in results]
        assert len(results) == 8
        return len(results) - len(set(predictions))

    by_question = lambda datum: datum["questionId"]

    def load_data_(model):
        data = [
            merge(datum, {"is-correct": is_correct(datum), "lang": lang})
            for lang in languages
            for datum in load_data(model, lang)
        ]
        data = sorted(data, key=by_question)
        return {k: list(group) for k, group in groupby(data, by_question)}

    data1 = load_data_("tt-mml-pretrain-epoch-22-5")
    data2 = load_data_("tt-mml-pretrain-epoch-22-finetune")

    data = [(data1[k], data2[k]) for k in data1.keys()]
    data = sorted(data, key=lambda d: agreement(d[1]) - agreement(d[0]), reverse=True)
    # data = sorted(data, key=lambda d: agreement(d[1]) + agreement(d[0]), reverse=False)

    # import random
    # random.shuffle(data)

    # data = [datum for datum in data if datum[0][0]["questionId"] == "201247133" or datum[0][0]["imageId"] in {"n119944", "n90294", "n146555", "n435808", "n222915"}] + data

    # Selected images and questions in the paper...
    SELECTED_IGMS = "n393305 n83784 n435808 n379991".split()[2:]
    SELECTED_QS = [
        "What is located on top of the pole?",
        "Is the black and white cat unhappy or happy?",
        "What is sitting next to the computer mouse?",
        "Which kind of cooking utensil is flat?",
    ]
    data = [
        datum
        for datum in data
        # if datum[0][0]["imageId"] in SELECTED_IGMS
        # and any(datum[0][2]["question"].startswith(q) for q in SELECTED_QS)
    ]

    for datum1, datum2 in data[:32]:
        elem = first(e for e in datum1 if e["lang"] == "en")

        question_id = elem["questionId"]
        question = elem["question"]
        answer = elem["answer"]
        image_id = elem["imageId"]

        lang_pred1 = [
            {"model": "en", "lang": elem["lang"], "pred": elem["prediction"]}
            for elem in datum1
        ]
        lang_pred2 = [
            {"model": "tt", "lang": elem["lang"], "pred": elem["prediction"]}
            for elem in datum2
        ]
        df = pd.DataFrame(lang_pred1 + lang_pred2)
        df = df.replace(
            {
                "lang": {
                    "bn": "BEN",
                    "de": "DEU",
                    "en": "ENG",
                    "id": "IND",
                    "ko": "KOR",
                    "pt": "POR",
                    "ru": "RUS",
                    "zh": "CMN",
                }
            }
        )
        df = df.pivot("lang", "model", "pred")

        st.code(question_id)
        st.code(image_id)
        st.image("../../xGQA/images/{}.jpg".format(image_id))
        st.code(question)
        st.code(answer)
        df
        str_latex = (
            "\n".join(
                [
                    "\multicolumn{2}{c}{\includegraphics[height=3cm]{imgs/xgqa-images/"
                    + image_id
                    + ".jpg}}",
                    "\multicolumn{2}{c}{" + question + "}",
                    "\multicolumn{2}{c}{" + answer + "}",
                ]
            )
            + "\n"
            + "\n".join(df.to_latex().split("\n")[5:13])
        )
        st.code(str_latex)
        st.markdown("---")

    # pdb.set_trace()


def main():
    # ensemble_results("tt-mml-pretrain")
    # ensemble_results("tt-mml-pretrain-finetune")
    show_qualitative_agreement_increase()
    # agreement_across_languages()


if __name__ == "__main__":
    main()
