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

from translate_cc import PATH_DATA, MODEL_TYPES, load_data, load_model, save_data, translate


TARGET_LANGUAGES = "de zh ja fr cs".split()


@click.command(help="Translate the full Conceptual Captions (CC) dataset into a target language")
@click.option("-s", "--split", type=click.Choice(["train", "valid"]), help="data split")
@click.option("-l", "--language", type=click.Choice(TARGET_LANGUAGES), help="target language")
@click.option("-m", "--model", "model_type", type=click.Choice(MODEL_TYPES), help="translation model")
@click.option("--device")
@click.option("-v", "--verbose", is_flag=True)
def main(split, language, model_type, device="cuda", verbose=False):
    # load English data and cached translations from `folder_input`
    folder_input = model_type + "-seed-1337"
    folder_output = model_type + "-full"

    if os.path.exists(os.path.join(PATH_DATA, folder_output, f"{language}-{split}.json")):
        print(f"File exists for language {language} and split {split}.")
        print(f"Exiting...")
        sys.exit(1)

    data = load_data(split, folder_input)
    # model = load_model(model_type, device)

    if model_type == "m2m-100-sm":
        if socket.gethostname() == "tesla":
            batch_size = 16
        else:
            batch_size = 8
    else:
        if socket.gethostname() == "tesla":
            batch_size = 8
        else:
            batch_size = 4

    path_cache = os.path.join(PATH_DATA, folder_input, ".cache", f"{language}-{split}")
    with shelve.open(path_cache) as data_cached:
        # data_tr = translate(data, model, language, batch_size, data_cached, verbose=verbose)
        data_tr = {k: v for k, v in tqdm.tqdm(data_cached.items())}
        save_data(data_tr, split, language, folder_output)


if __name__ == "__main__":
    main()
