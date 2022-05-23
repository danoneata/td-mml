# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import logging

import tensorpack.dataflow as td

import numpy as np
import torch.distributed as dist

from volta.datasets.concept_cap_dataset import (
    ConceptCapLoaderTrain,
    ConceptCapLoaderVal,
    BertPreprocessBatch,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ConceptCapMultilingualLoaderTrain(ConceptCapLoaderTrain):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
        self,
        annotations_path,
        features_path,
        tokenizer,
        bert_model,
        seq_len,
        langs,
        langs_sampling_path=None,
        batch_size=512,
        num_workers=25,
        cache=10000,
        local_rank=-1,
        objective=0,
        num_locs=5,
        add_global_imgfeat=None,
        tokenizer_name="bert-base-uncased",
        lmdb_file = None
    ):
        if lmdb_file is None:
            if dist.is_available() and local_rank != -1:
                rank = dist.get_rank()
                lmdb_file = os.path.join(features_path, "training_feat_part_" + str(rank) + ".lmdb")
            else:
                lmdb_file = os.path.join(features_path, "training_feat_all.lmdb")

                print("Loading from %s" % lmdb_file)

            ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        else:
            ds = lmdb_file
        self.num_dataset = len(ds)
        ds = td.LocallyShuffleData(ds, cache)

        preprocess_function = BertPreprocessMultilingualBatch(
            annotations_path,
            tokenizer,
            bert_model,
            seq_len,
            36,
            self.num_dataset,
            langs=langs,
            langs_sampling_path=langs_sampling_path,
            objective=objective,
            num_locs=num_locs,
            tokenizer_name=tokenizer_name,
            split="train",
        )

        ds = td.PrefetchData(ds, 16, 1)
        ds = td.MapData(ds, preprocess_function)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_global_imgfeat = add_global_imgfeat
        self.num_locs = num_locs


class ConceptCapMultilingualLoaderVal(ConceptCapLoaderVal):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
            self,
            annotations_path,
            features_path,
            tokenizer,
            bert_model,
            seq_len,
            langs,
            langs_sampling_path=None,
            batch_size=512,
            num_workers=25,
            cache=5000,
            objective=0,
            num_locs=5,
            add_global_imgfeat=True,
            tokenizer_name="bert-base-uncased",
            visualization=False,
    ):
        lmdb_file = os.path.join(features_path, "validation_feat_all.lmdb")
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessMultilingualBatch(
            annotations_path,
            tokenizer,
            bert_model,
            seq_len,
            36,
            self.num_dataset,
            langs=langs,
            langs_sampling_path=langs_sampling_path,
            visualization=visualization,
            objective=objective,
            num_locs=num_locs,
            tokenizer_name=tokenizer_name,
            split="valid",
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_global_imgfeat = add_global_imgfeat
        self.num_locs = num_locs


class BertPreprocessMultilingualBatch(BertPreprocessBatch):
    def __init__(
            self,
            caption_path,
            tokenizer,
            bert_model,
            seq_len,
            region_len,
            data_size,
            langs,
            langs_sampling_path=None,
            split="Train",
            visualization=False,
            objective=0,
            num_locs=5,
            tokenizer_name="bert-base-uncased",
    ):

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.num_caps = data_size
        self.langs = langs
        self.captions = self.load_captions(caption_path, split, self.langs)
        self.visualization = visualization
        self.objective = objective
        self.bert_model = bert_model
        self.num_locs = num_locs
        self.tokenizer_name = tokenizer_name
        self.langs_sampling_path = langs_sampling_path
        self.sample_language = self.prepare_langauge_sampler(langs_sampling_path)

    def prepare_langauge_sampler(self, path):
        if path is None or path == "":
            return LanguageSamplingDefault(self.captions)

        elif os.path.exists(path):
            return LanguageSamplingGiven(path, self.langs)
        else:
            assert False, "path should be `None` or exist:\n{}".format(path)

    @staticmethod
    def load_captions(path, split, langs):
        def load1(lang):
            with open(os.path.join(path, f"{lang}-{split}.json")) as f:
                return json.load(f)
        return {lang: load1(lang) for lang in langs}

    def random_cap(self, caption, image_id):
        if not self.visualization and self.objective != 2 and random.random() > 0.5:
            # Pick a random caption from a random language.
            lang = random.choice(self.langs)
            caption = random.choice(list(self.captions[lang].values()))
            label = 1
        else:
            # Pick the correpsonding caption of the current image from a random
            # language for which we have the translation.

            lang = self.sample_language(image_id)
            caption = self.captions[lang][image_id]
            label = 0

        return caption, label


# Functions used for language sampling

class LanguageSamplingDefault:
    """Uniformly picks one of the languages for which we have a translation."""
    def __init__(self, captions):
        self.captions = captions

    def __call__(self, image_id):
        langs_translated = [
            lang
            for lang, captions_lang in self.captions.items()
            if image_id in captions_lang
        ]
        return random.choice(langs_translated)

class LanguageSamplingGiven:
    """Picks a language based on the probabilities specified at the given path.
    This function is used to replicate the adjusted langauge probabilities of
    Conneau and Lample, but using our sampling procedure which first samples
    the ids and then the languages. See the following notebook for more
    details:

    https://colab.research.google.com/drive/1bH3vyF6YhniM7XVXyIHoiDpHN57Kth1O

    """
    def __init__(self, path, langs_=None):
        data = np.load(path)
        p_lang_and_sent = data["p_lang_and_sent"]
        langs = data["langs"]
        sents = data["sents"]
        # compute p(lang | sent)
        p_sent = p_lang_and_sent.sum(axis=0, keepdims=True)
        p_lang_given_sent = p_lang_and_sent / p_sent
        # data needed by `__call__`

        self.langs = langs
        self.p_lang_given_sent = {
            sent: p_lang_given_sent[:, i]
            for i, sent in enumerate(sents)
        }
        self.langs_ = langs_

    def __call__(self, image_id):
        p = self.p_lang_given_sent[image_id]
        if self.langs_ is None:
            return random.choices(self.langs, weights=p)[0]
        else:
            out_langs = None
            while out_langs not in self.langs_:
                out_langs = random.choices(self.langs, weights=p)[0]
            return out_langs
