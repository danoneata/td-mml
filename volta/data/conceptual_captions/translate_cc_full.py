import csv
import json
import os
import pdb
import random
import shelve
import sys

from typing import Dict, List

import click
import tqdm

from toolz import first, partition_all

from easynmt import EasyNMT

from translate_cc import (
    PATH_DATA,
    MODEL_TYPES,
    get_batch_size,
    load_data,
    load_model,
    save_data,
    translate,
)


def read(path, parse=lambda line: line.strip()):
    with open(path, "r") as f:
        return list(map(parse, f.readlines()))


# TARGET_LANGUAGES = "de zh ja fr cs".split()  # UC2 languages
# TARGET_LANGUAGES = "id sw ta tr zh".split()  # MaRVL languages
TARGET_LANGUAGES = read("data/langs-iglue.txt")


@click.command(help="Translate the full Conceptual Captions (CC) dataset into a target language")
@click.option("-s", "--split", type=click.Choice(["train", "valid"]), help="data split")
@click.option("-l", "--language", type=click.Choice(TARGET_LANGUAGES), help="target language")
@click.option("-m", "--model", "model_type", type=click.Choice(MODEL_TYPES), help="translation model")
@click.option("-b", "--batch-size", "batch_size", type=click.INT, default=None, help="if not set, estimated based on the machine")
@click.option("-k", "--path-keys", "path_keys", help="json mapping a language to the keys that we need to translate")
@click.option("--device")
@click.option("-v", "--verbose", is_flag=True)
def main(split, language, model_type, batch_size=None, path_keys=None, device="cuda", verbose=False):
    # load English data and cached translations from `folder_input`
    folder_input = model_type + "-seed-1337"
    folder_output = model_type + "-full"

    if os.path.exists(os.path.join(PATH_DATA, folder_output, f"{language}-{split}.json")):
        print(f"File exists for language {language} and split {split}.")
        print(f"Exiting...")
        sys.exit(1)

    data = load_data(split, "en", folder_input)
    model = load_model(model_type, device)
    batch_size = batch_size or get_batch_size(model_type, device)

    if path_keys:
        with open(path_keys, "r") as f:
            lang_to_keys = json.load(f)
            keys = lang_to_keys[languages]
    else:
        keys = data.keys()

    path_cache = os.path.join(PATH_DATA, folder_input, ".cache", f"{language}-{split}")
    with shelve.open(path_cache) as data_cached:
        data_tr = translate(data, model, language, batch_size, data_cached, keys=keys, verbose=verbose)
        # data_tr = {k: v for k, v in tqdm.tqdm(data_cached.items())}
        save_data(data_tr, split, language, folder_output)


if __name__ == "__main__":
    main()
