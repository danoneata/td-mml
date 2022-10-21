# TD-MML: Translated Data for Multilingual Multimodal Learning

This repository contains the implementation for the paper:

> Chen Qiu, Dan Oneață, Emanuele Bugliarello, Stella Frank, Desmond Elliott.
> Multilingual Multimodal Learning with Machine Translated Text.
> EMNLP, 2022.

## Setup

This repository is a fork of the [IGLUE](https://github.com/e-bug/iglue) codebase, which in turn depends on [VOLTA](https://github.com/e-bug/volta).
To set up the environment, please follow the instructions listed in the [VOLTA README](https://github.com/e-bug/iglue/blob/main/volta/README.md).

## Data

[`datasets/`](datasets) contains the textual data for each dataset.

Check out its [README](datasets/README.md) for links to preprocessed data.

Features extraction steps for each of dataset and backbone can be found under [`features_extraction/`](features_extraction). 

## Models

The checkpoints of all the pretrained V&L models can be downloaded from [ERDA](https://sid.erda.dk/sharelink/b1Rge0DwwW).

For more details on defining new models in VOLTA, see [`volta/MODELS.md`](volta/MODELS.md).

Model configuration files are stored in [`volta/config/`](volta/config). 


## Training and Evaluation

We provide the scripts we used to train and evaluate models in [`experiments/`](experiments):
- [`zero_shot/`](experiments/zero_shot): English fine-tuning and zero-shot/`translate test' evaluation
- [`few_shot/`](experiments/few_shot): Few-shot experiments for each dataset-language-shots triplet
- [`few_shot.dev-mt/`](experiments/few_shot.dev-mt): Few-shot experiments when using dev sets in the target languages (MT)
- [`translate_train.de/`](experiments/translate_train.de): `Translate train' experiments on xFLickr&CO in German
- [`translate_train.ja/`](experiments/translate_train.ja): `Translate train' experiments on xFLickr&CO in Japanese

Task configuration files are stored in [config_tasks/](config_tasks).

## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data are subject to their respective licenses.

If you find our code/data/models or ideas useful in your research, please consider citing the paper.
