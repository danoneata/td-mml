import json
import sys
import pdb
import pickle
import os

from typing import Any, List, Dict, TypedDict

import click
import shelve

from tqdm import tqdm
from toolz import merge

from translate_cc_full import load_data


# def load_data(lang, version):
#     with open(f"data/cc/analysis-cache-v{version:03d}/{lang}.json", "r") as f:
#         return json.load(f)


BASE_PATH = "data/cc/analysis-cache"
Scored = TypedDict("Scored", {"key": str, "score": float})


def save_data(data: Dict, lang: str, subset: bool) -> None:
    suffix = "-subset" if subset else ""
    with open(f"data/cc/analysis-cache/{lang}{suffix}.json", "w") as f:
        json.dump(data, f, indent=4)


def compute_lm_score(data):
    from lm_scorer.models.auto import AutoLMScorer as LMScorer

    device = "cuda:5"
    batch_size = 6

    sentences = [datum["text-src"] for datum in data][:10]
    scorer = LMScorer.from_pretrained("gpt2", device=device, batch_size=batch_size)
    scores = scorer.sentence_score(sentences, log=True)
    pdb.set_trace()


def compute_uniformity(data: Dict, cache) -> List[Scored]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    @shelve_cache(cache)
    def do1(datum):
        tokens = tokenizer(datum["text-tgt"])["input_ids"]
        unique = set(tokens)  # unique tokens
        return 1 - len(unique) / len(tokens)

    return [
        {
            "key": datum["key"],
            "score": do1(datum),
        }
        for datum in tqdm(data)
    ]


def compute_len_ratio(data: Dict) -> List[Scored]:
    pass


def compute_sim_tgt_src(data: Dict) -> List[Scored]:
    pass


def compute_bleu_translated(data: Dict, cache) -> List[Scored]:
    from nltk.translate.bleu_score import sentence_bleu

    @shelve_cache(cache)
    def do1(datum):
        return sentence_bleu([datum["text-src"]], datum["text-tgt"])

    return [
        {
            "key": datum["key"],
            "score": do1(datum),
        }
        for datum in tqdm(data)
    ]


# def cache(func, name):
#     path = os.path.join(BASE_PATH, ".cache", name + ".pkl")
#     def wrapped(*args, **kwargs):
#         if os.path.exists(path):
#             with open(path, "rb") as f:
#                 return pickle.load(f)
#         else:
#             res = func(*args, **kwargs)
#             with open(path, "wb") as f:
#                 pickle.dump(res, f)
#             return res
# 
#     return wrapped


def shelve_cache(cache):
    def wrapped(func):
        def inner(datum):
            key = datum["key"]
            try:
                return cache[key]
            except KeyError:
                result = func(datum)
                cache[key] = result
                return result
        return inner
    return wrapped


def merge_features(feature_data, data):
    features = feature_data.keys()
    return [
        merge(data[i], {"scores": {f: feature_data[f][i]["score"] for f in features}})
        for i in range(len(data))
    ]


FEATURES = {
    "uniformity": compute_uniformity,
    "len-ratio": compute_len_ratio,
    "sim-tgt-src": compute_sim_tgt_src,
    "sim-tgt-src-bleu": compute_bleu_translated,
}



LANGUAGES = "ar bg bn da de el es et fr id ja ko pt ru sw ta tr vi zh".split()


@click.command(help="Extract translation features")
@click.option(
    "-l",
    "--lang",
    type=click.Choice(LANGUAGES),
    help="which language to analyze",
)
@click.option(
    "-f",
    "--features",
    type=click.Choice(FEATURES),
    multiple=True,
    help="name of features",
)
@click.option(
    "-s",
    "--subset",
    is_flag=True,
    default=False,
    help="compute only on part of the keys",
)
def main(lang: str, features: List[str], subset: bool) -> None:

    split = "train"
    folder = "m2m-100-lg-full"

    data_src = load_data(split, "en", folder)
    data_tgt = load_data(split, lang, folder)

    if subset:
        with open(os.path.join(BASE_PATH, "keys.json"), "r") as f:
            keys = json.load(f)
    else:
        keys = list(data_src.keys())

    data = [
        {
            "key": key,
            "text-src": data_src[key],
            "text-tgt": data_tgt[key],
        }
        for key in tqdm(keys)
    ]

    def get_shelve_path(lang: str, feat: str) -> str:
        return os.path.join(BASE_PATH, ".cache-shelve", lang + "-" + feat)

    feature_data = {}
    for feat in features:
        with shelve.open(get_shelve_path(lang, feat)) as cache:
            feature_data[feat] = FEATURES[feat](data, cache)

    # feature_data = {feat: cache(FEATURES[feat], lang + "-" + feat)(data) for feat in features}
    feature_data = merge_features(feature_data, data)
    save_data(feature_data, lang, subset)


if __name__ == "__main__":
    main()
