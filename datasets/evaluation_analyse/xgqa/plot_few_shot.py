import os
import pdb

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib import rcParams

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


sns.set_theme("talk", font_scale=1.4)


def read_txt(path, file_name):
    xGQA_data = [
        ln.strip().split("\t") for ln in open(os.path.join(path, file_name)).readlines()
    ]
    return xGQA_data


def draw_avgs():
    modes = ["zero_shot", "few_shot", "translate_train"]
    strategy = ["xuniter", "random_lg", "vtlm"]
    types = ["compare", "choose", "logical", "query", "verify"]
    langs = ["bn", "de", "en", "id", "ko", "pt", "ru", "zh"]
    # x_plots = ['zero_shot', '1-shot', '5-shot', '10-shot', '20-shot', '25-shot', '48-shot', 'multi-eval']
    x_plots = ["0", "1", "5", "10", "20", "25", "48", "multi"]
    dir_path = "xgqa_data"

    files_dict = {}
    for mode in modes:
        path = os.path.join(dir_path, mode)
        for strategy_ in strategy:
            if mode == "translate_train":
                file_name = f"test.{strategy_}.8langs.{mode}_filter.txt"
            else:
                file_name = f"test.{strategy_}.8langs.{mode}.txt"
            files_dict[str(mode + "-" + strategy_)] = read_txt(path, file_name)

    xUNITER = [[[] for _ in range(len(types))] for i in range(len(langs))]
    random_lg = [[[] for _ in range(len(types))] for i in range(len(langs))]
    vtlm = [[[] for _ in range(len(types))] for i in range(len(langs))]
    all_data = [xUNITER, random_lg, vtlm]
    # model
    # lang
    # type
    # shots

    data_frame = []

    for k, strategy_ in enumerate(strategy):
        for i, lang in enumerate(langs):
            for j, type_ in enumerate(types):
                for l, mode in enumerate(modes):
                    data_in_ = files_dict[str(mode + "-" + strategy_)]
                    if mode in ["few_shot"]:
                        for x_id, x in enumerate(["1", "5", "10", "20", "25", "48"]):
                            result_type = data_in_[i * (3 * 6) + 3 * x_id + 2]
                            all_data[k][i][j].append(result_type[j])
                            datum = {
                                "model": strategy_,
                                "lang": lang,
                                "type": type_,
                                "finetune": mode,
                                "shots": x_id + 1,
                                "accuracy": float(result_type[j]),
                            }
                            data_frame.append(datum)

                    else:
                        result_type = data_in_[i * 3 + 2]
                        all_data[k][i][j].append(result_type[j])
                        datum = {
                            "model": strategy_,
                            "lang": lang,
                            "type": type_,
                            "finetune": "few_shot" if mode == "zero_shot" else mode,
                            "shots": 0 if mode == "zero_shot" else None,
                            "accuracy": float(result_type[j]),
                        }
                        data_frame.append(datum)

    df = pd.DataFrame(data_frame)
    df = df[df["model"] != "random_lg"]

    def replicate_zero_shot(row):
        if row.finetune == "translate_train":
            row.shots = np.arange(0, 7)
        return row

    df1 = df.apply(replicate_zero_shot, axis=1).explode("shots", ignore_index=True)
    df1

    df1 = df1.replace({
        "model": {"xuniter": "xUNITER", "vtlm": "TD-MML"},
        "finetune": {"few_shot": "zero & few shot", "translate_train": "translated data"},
        "shots": dict(enumerate(["0", "1", "5", "10", "20", "25", "48"])),
    })

    fig = sns.relplot(
        data=df1,
        x="shots",
        y="accuracy",
        col="type",
        hue="model",
        style="finetune",
        kind="line",
    )
    # sns.move_legend(fig, "lower right")

    fig.set_xlabels("num. shots")
    fig.set_ylabels("accuracy (%)")

    fig.savefig("output/few-shot/xgqa-few-shot.pdf")
    st.pyplot(fig)


def main():
    # draw_per_lang()
    draw_avgs()


if __name__ == "__main__":
    main()
