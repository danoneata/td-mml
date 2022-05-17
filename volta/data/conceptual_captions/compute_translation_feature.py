import json
import sys
import pdb
import os

from tqdm import tqdm


def load_data(lang, version):
    with open(f"data/cc/analysis-cache-v{version:03d}/{lang}.json", "r") as f:
        return json.load(f)


def save_data(data, lang, version):
    new_version = version + 1
    os.makedirs(f"data/cc/analysis-cache-v{new_version:03d}", exist_ok=True)
    with open(f"data/cc/analysis-cache-v{new_version:03d}/{lang}.json", "w") as f:
        json.dump(data, f, indent=4)


def compute_lm_score(data):
    from lm_scorer.models.auto import AutoLMScorer as LMScorer

    device = "cuda:5"
    batch_size = 6

    sentences = [datum["text-src"] for datum in data][:10]
    scorer = LMScorer.from_pretrained("gpt2", device=device, batch_size=batch_size)
    scores = scorer.sentence_score(sentences, log=True)
    pdb.set_trace()


def compute_blue_translated(data):
    from nltk.translate.bleu_score import sentence_bleu

    for datum in tqdm(data):
        datum["scores"]["sim-tgt-src-bleu"] = sentence_bleu(
            [datum["text-src"]], datum["text-tgt"]
        )

    return data


def main():
    lang = sys.argv[1]
    version = int(sys.argv[2])

    assert lang in "id sw ta tr zh".split()

    data = load_data(lang, version)
    # data_updated = compute_lm_score(data)
    data_updated = compute_blue_translated(data)

    save_data(data_updated, lang, version)


if __name__ == "__main__":
    main()
