import csv
import json
import os
import pdb
import random
import shelve
import socket
import sys

from typing import Dict, List

import click
import tqdm

from toolz import first, partition_all

from easynmt import EasyNMT


PATH_DATA = "data/cc/translations"


def load_language_counts() -> List[Dict]:
    def parse(row):
        code, lang, _, count = row
        code = code.strip()
        lang = lang.strip()
        count = int(count.split(",")[0])
        return {"code": code, "lang": lang, "count": count}

    with open("data/xmlr-counts.csv", "r") as f:
        reader = csv.reader(f, delimiter=";")
        _ = next(reader)
        data = [parse(row) for row in reader]
        data = [d for d in data if d["code"] not in {"", "-"}]
        return data


MODEL_TYPES = ["m2m-100-md", "m2m-100-lg"]
LANG_COUNTS = load_language_counts()
TARGET_LANGUAGES = [d["code"] for d in LANG_COUNTS if d["code"] != "en"]


def load_data(split, language, folder):
    path = os.path.join(PATH_DATA, folder, f"{language}-{split}.json")
    with open(path, "r") as f:
        return json.load(f)


def save_data(data, split, language, folder):
    path = os.path.join(PATH_DATA, folder, f"{language}-{split}.json")
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def load_model(model_type, device):
    if model_type == "m2m-100-md":
        return EasyNMT("m2m_100_418M", device=device)
    elif model_type == "m2m-100-lg":
        return EasyNMT("m2m_100_1.2B", device=device)
    else:
        assert False


def translate(data, model, language, batch_size, data_cached, count_target=None, keys=None, verbose=0):
    keys = keys or data.keys()

    count = 0
    data_tr = dict()
    to_break = False

    for keys in partition_all(batch_size, keys):

        if all(key in data_cached for key in keys):
            sentences = [data[key] for key in keys]
            sentences_tr = [data_cached[key] for key in keys]
            to_cache = False
        else:
            sentences = [data[key] for key in keys]
            sentences_tr = model.translate_sentences(
                sentences,
                target_lang=language,
                source_lang="en",
                batch_size=batch_size,
            )
            to_cache = True

        for key, sentence_tr in zip(keys, sentences_tr):
            data_tr[key] = sentence_tr

            if to_cache:
                data_cached[key] = sentence_tr

            count = count + len(sentence_tr)

            if count_target and count > count_target:
                to_break = True
                break

        if to_break:
            break

        if verbose:
            count_target_str = "{:9d}".format(count_target) if count_target else "∞"
            print("{:9d} / {}".format(count, count_target_str))
            for k, s, t in zip(keys, sentences, sentences_tr):
                print(k)
                print(s)
                print(t)
                print()
            pdb.set_trace()
        else:
            count_target_str = "{:9d}".format(count_target) if count_target else "∞"
            print(
                "{} · {:12s} · {:9d} / {}".format(language, key, count, count_target_str),
                end="\r",
                flush=True,
            )

    return data_tr


@click.command(help="Translate the Conceptual Captions (CC) dataset into a target language")
@click.option("-s", "--split", type=click.Choice(["train", "valid"]), help="data split")
@click.option("-l", "--language", type=click.Choice(TARGET_LANGUAGES), help="target language")
@click.option("-m", "--model", "model_type", type=click.Choice(MODEL_TYPES), help="translation model")
@click.option("-r", "--random-seed", "seed", default=1337, type=click.INT)
@click.option("--device")
@click.option("-v", "--verbose", is_flag=True)
def main(split, language, model_type, seed, device="cuda", verbose=False):
    folder = model_type + "-seed-" + "{:04d}".format(seed)

    if os.path.exists(os.path.join(PATH_DATA, folder, f"{language}-{split}.json")):
        print(f"File exists for language {language} and split {split}.")
        print(f"Exiting...")
        sys.exit(1)

    data = load_data(split, "en", folder)
    model = load_model(model_type, device)

    keys = list(data.keys())

    if split == "train" and model_type == "m2m-100-md":
        # update base seed to have different sentences sampled for each language
        seed = seed + TARGET_LANGUAGES.index(language)
        random.seed(seed)
        random.shuffle(keys)
        count_target = first(d["count"] for d in LANG_COUNTS if d["code"] == language)
    elif split == "train" and model_type == "m2m-100-lg":
        # use the same keys as the m2m-100-md model
        data_md = load_data(split, language, f"m2m-100-md-seed-{seed:04d}")
        keys = list(data_md.keys())
        count_target = None
    else:
        # for validation translate all sentences
        count_target = None

    if model_type == "m2m-100-md":
        if socket.gethostname() == "tesla":
            batch_size = 16
        else:
            batch_size = 8
    else:
        if socket.gethostname() == "tesla":
            batch_size = 8
        else:
            batch_size = 4

    path_cache = os.path.join(PATH_DATA, folder, ".cache", f"{language}-{split}")
    with shelve.open(path_cache) as data_cached:
        data_tr = translate(
            data,
            model,
            language,
            batch_size,
            data_cached,
            count_target=count_target,
            keys=keys,
            verbose=verbose,
        )
    save_data(data_tr, split, language, folder)


if __name__ == "__main__":
    main()
