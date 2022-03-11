# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import logging

import tensorpack.dataflow as td

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
        batch_size=512,
        num_workers=25,
        cache=10000,
        local_rank=-1,
        objective=0,
        num_locs=5,
        add_global_imgfeat=None,
        tokenizer_name="bert-base-uncased",
    ):

        if dist.is_available() and local_rank != -1:
            rank = dist.get_rank()
            lmdb_file = os.path.join(features_path, "training_feat_part_" + str(rank) + ".lmdb")
        else:
            lmdb_file = os.path.join(features_path, "training_feat_all.lmdb")

            print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
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
            langs_translated = [
                lang
                for lang, captions_lang in self.captions.items()
                if image_id in captions_lang
            ]
            lang = random.choice(langs_translated)
            caption = self.captions[lang][image_id]
            label = 0
        return caption, label
