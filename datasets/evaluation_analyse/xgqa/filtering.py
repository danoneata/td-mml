import json
import os
import pdb
import random

import pandas as pd
import streamlit as st
import seaborn as sns

from matplotlib import pyplot as plt
from toolz import assoc, merge


random.seed(1337)
st.set_page_config(layout="wide")
sns.set_theme("talk")


def load_scored_data(lang):
    with open(f"../tt-mml-www/data/cc/analysis-{lang}-cache.json") as f:
        return json.load(f)


langs = (
    "ar bg bn da de el es et fr id ja ko pt ru sw ta tr vi zh".split()
)  # IGLUE languages
lang_to_iso3 = {
    "ar": "ARB",
    "bg": "BEG",
    "bn": "BUL",
    "da": "DAN",
    "de": "DEU",
    "el": "ELL",
    "es": "EST",
    "et": "EST",
    "fr": "FRA",
    "id": "IND",
    "ja": "JPN",
    "ko": "KOR",
    "pt": "POR",
    "ru": "RUS",
    "sw": "SWA",
    "ta": "TAM",
    "tr": "TUR",
    "vi": "VIE",
    "zh": "CMN",
}

u = 0.5
s_script = 0.1
s_non_indo = 0.5
s_indo = 0.7

groups = {
    "script": "ar bg bn el ja ko ru ta zh".split(),
    "non-indo": "et id sw tr vi".split(),
    "indo": "da de fr es pt".split(),
}

lang_to_group = {l: g for g, ls in groups.items() for l in ls}

threshs = {
    "ar": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "bg": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "bn": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "da": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "de": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "el": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "es": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "et": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "fr": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "id": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "ja": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "ko": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "pt": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "ru": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "sw": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "ta": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "tr": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "vi": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "zh": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
}


df = pd.DataFrame(
    [
        merge(datum["scores"], {"lang": lang, "group": lang_to_group[lang]})
        for lang in langs
        for datum in load_scored_data(lang)
    ]
)

df


sns.set_palette("colorblind")
# individual languages
for i, (name, langs) in enumerate(groups.items()):
    i = 0

    fig, axs = plt.subplots(
        ncols=2, nrows=1, figsize=(2 * 6.4, 0.7 * 4.8), squeeze=False, sharey=True
    )
    df1 = df[df["lang"].isin(langs)]
    df1 = df1.replace({"lang": lang_to_iso3})
    df1 = df1.sort_values("lang")

    τ = threshs[langs[0]]
    axs[i, 0].vlines(τ["uniformity"], 0, 1, color="gray")
    axs[i, 1].vlines(τ["sim-tgt-src-bleu"], 0, 1, color="gray")

    axs[i, 0].set_xlim([0, 1])
    axs[i, 1].set_xlim([0, 1])

    axs[i, 0].set_xlabel("1 - TTR")
    axs[i, 1].set_xlabel("BLEU src-tgt")

    axs[i, 0].set_ylabel("fraction")
    # axs[i, 0].set_ylabel("fraction (log)")
    # axs[i, 1].set_yscale("log")

    sns.ecdfplot(df1, x="uniformity", hue="lang", ax=axs[i, 0])
    sns.ecdfplot(df1, x="sim-tgt-src-bleu", hue="lang", ax=axs[i, 1])

    sns.move_legend(axs[i, 0], "lower right", title="", fontsize=12)
    sns.move_legend(axs[i, 1], "lower right", title="", fontsize=12)

    st.pyplot(fig)

    # plt.tight_layout()
    plt.savefig(f"output/filtering/filtering-{name}.pdf", bbox_inches="tight")

# fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(2 * 6.4, 1 * 4.8), squeeze=False)
# i = 0
# df1 = df
#
# sns.set_theme()
# sns.ecdfplot(df1, x="uniformity", hue="group", ax=axs[i, 0])
# sns.ecdfplot(df1, x="sim-tgt-src-bleu", hue="group", ax=axs[i, 1])
#
# sns.move_legend(axs[i, 0], "lower right")
# sns.move_legend(axs[i, 1], "lower right")
#
# τ = threshs[langs[0]]
# axs[i, 0].vlines(τ["uniformity"], 0, 1, color="gray")
# axs[i, 1].vlines(τ["sim-tgt-src-bleu"], 0, 1, color="gray")
#
# axs[i, 0].set_xlim([0, 1])
# axs[i, 1].set_xlim([0, 1])
#
# axs[i, 0].set_xlabel("1 - TTR")
# axs[i, 0].set_ylabel("fraction")
#
# axs[i, 1].set_xlabel("BLEU src-tgt")
# axs[i, 1].set_ylabel("fraction")
#
# st.pyplot(fig)
# # plt.tight_layout()
# plt.savefig(f"output/filtering/filtering-{name}.png")
