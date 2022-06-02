import json


def load_data(lang):
    with open(f"data/cc/analysis-cache/{lang}.json", "r") as f:
        return json.load(f)


def save_data(data, lang):
    with open(f"data/cc/translations/m2m-100-lg-filtered/{lang}.json", "w") as f:
        json.dump(data, f, ensure_ascii=False)


u = 0.5
s_script = 0.1
s_non_indo = 0.5
s_indo = 0.7

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
        "sim-tgt-src-bleu": s_indo,
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


langs = threshs.keys()
metrics = threshs["ar"].keys()


def is_valid(datum, lang):
    return all(datum["scores"][m] <= threshs[lang][m] for m in metrics)


def main():
    for lang in langs:
        data1 = load_data(lang)
        data2 = {datum["key"]: datum["text-tgt"] for datum in data1 if is_valid(datum, lang)} 
        print("{} · {} → {} ({:.2%})".format(lang, len(data1), len(data2), len(data2) / len(data1)))
        save_data(data2, lang)


if __name__ == "__main__":
    main()
