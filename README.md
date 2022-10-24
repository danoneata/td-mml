# TD-MML: Translated Data for Multilingual Multimodal Learning

This repository contains the implementation for the paper:

> Chen Qiu, Dan Oneață, Emanuele Bugliarello, Stella Frank, Desmond Elliott.
> _Multilingual Multimodal Learning with Machine Translated Text._
> EMNLP, 2022.

## Setup

This repository is a fork of the [IGLUE](https://github.com/e-bug/iglue) codebase, which in turn depends on [VOLTA](https://github.com/e-bug/volta).
To set up the environment, please follow the instructions listed in the [VOLTA README](https://github.com/e-bug/iglue/blob/main/volta/README.md).

## Data

The machine translated data corresponding to the Conceptual Captions dataset can be downloaded from [here](https://sharing.speed.pub.ro/owncloud/remote.php/webdav/cc-translations-m2m-100-lg-iglue-languages-filtered.zip).
The Conceptual Captions datasets contains 2.77M English sentences gathered from web-crawled alt-text and post-processed to remove proper names.
We translated those sentences using the large M2M-100 model (with 1.2B parameters) into the twenty languages of the IGLUE benchmark.
Since we have observed that the quality translations varies across languages, we have applied an automatic filtering procedure to discard poor sentences (see the paper for more details);
the provided data contains the filtered translations.

We also provide translations for two of the IGLUE tasks in two variants (filtered and full):
- MaRVL (based on the NLVR2 dataset): [filtered translations](https://sharing.speed.pub.ro/owncloud/index.php/s/2J4mLWncB1lEbGc) · [full translations](https://sharing.speed.pub.ro/owncloud/index.php/s/Ge2qATV3LLA7yME)
- xGQA (based on the GQA dataset): [filtered translations](https://sharing.speed.pub.ro/owncloud/index.php/s/NowXwNATWMApQRu) · [full translations](https://sharing.speed.pub.ro/owncloud/index.php/s/fHHiNOhdI1IMqi1)

The code to generate the translations is available in `volta/data/conceptual_captions`;
see the corresponding [README](https://github.com/danoneata/td-mml/tree/main/volta/data/conceptual_captions#translate-all-captions-in-the-iglue-languages-using-the-large-m2m-translation-model).

The visual features are the same as those used in IGLUE;
see the extraction steps for each of dataset and backbone under [`features_extraction/`](features_extraction).

## Models

The checkpoints of all the pretrained TD-MML model will be made available shortly.

For more details on defining new models in VOLTA, see [`volta/MODELS.md`](volta/MODELS.md).

Model configuration files are stored in [`volta/config/`](volta/config).

## Training and Evaluation

We provide the scripts we used to train and evaluate models in [`experiments/`](experiments):
- [`zero_shot/`](experiments/zero_shot): English fine-tuning and zero-shot/`translate test' evaluation
- [`few_shot/`](experiments/few_shot): Few-shot experiments for each dataset-language-shots triplet

Task configuration files are stored in [config_tasks/](config_tasks).

## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details.
Third-party software and data are subject to their respective licenses.

If you find our code/data/models or ideas useful in your research, please consider citing the paper.
