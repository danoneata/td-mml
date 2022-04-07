# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import logging
import numpy as np
import torch
import tensorpack.dataflow as td
import lmdb
from copy import copy

import torch.distributed as dist

from volta.datasets.concept_cap_dataset import (
    ConceptCapLoaderTrain,
    ConceptCapLoaderVal,
    BertPreprocessBatch,
    InputFeatures
)

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
            (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
            (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
            np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
            - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
            + 1
    )
    iw[iw < 0] = 0

    ih = (
            np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
            - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
            + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ConceptCap_itm(ConceptCapLoaderTrain):
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
    ):

        if dist.is_available() and local_rank != -1:
            rank = dist.get_rank()
            lmdb_file = os.path.join(features_path, "training_feat_part_" + str(rank) + ".lmdb")
        else:
            # lmdb_file = os.path.join(features_path, "training_feat.lmdb","training_feat_all.lmdb")
            lmdb_file = os.path.join(features_path, "training_feat_all.lmdb")

            print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        ds_preprocess = copy(ds)

        self.num_dataset = len(ds)
        ds = td.LocallyShuffleData(ds, cache)

        preprocess_function = BertPreprocessBatch_itm(
            annotations_path,
            ds_preprocess,
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


class ConceptCapVal_itm(ConceptCapLoaderVal):
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
        lmdb_file = os.path.join(features_path,"validation_feat.lmdb", "validation_feat_all.lmdb")
        # lmdb_file = os.path.join(features_path, "validation_feat_all.lmdb")
        print("Loading from %s" % lmdb_file)


        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        # preprocess_function = BertPreprocessMultilingualBatch(
        #     annotations_path,
        #     tokenizer,
        #     bert_model,
        #     seq_len,
        #     36,
        #     self.num_dataset,
        #     langs=langs,
        #     langs_sampling_path=langs_sampling_path,
        #     visualization=visualization,
        #     objective=objective,
        #     num_locs=num_locs,
        #     tokenizer_name=tokenizer_name,
        #     split="valid",
        # )
        #
        # ds = td.MapData(ds, preprocess_function)
        # self.ds = td.BatchData(ds, batch_size)
        # self.ds.reset_state()
        #
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.add_global_imgfeat = add_global_imgfeat
        # self.num_locs = num_locs


class BertPreprocessBatch_itm(BertPreprocessBatch):
    def __init__(
            self,
            caption_path,
            ds_preprocess,
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
        self.sample_language = self.prepare_langauge_sampler(self.langs_sampling_path)
        logger.info("  sample_language = %s", self.sample_language)
        self.ds_preprocess = ds_preprocess



    def __call__(self, data):
        image_feature_wp, image_cls_wp, obj_labels, obj_confs, attr_labels, attr_confs, attr_scores, \
        image_location_wp, num_boxes, image_h, image_w, image_id, caption = data

        update_caption, sample_caption, neg_lang, label, image_id, img = self.random_cap(caption, image_id=image_id,
                                                                                         neg_img=0.2,neg_lang1=0.4, neg_lang2=0.4)

        #get randomly sampled image with img
        if img:
            image_feature_wp, image_cls_wp, obj_labels, obj_confs, attr_labels, attr_confs, attr_scores, \
            image_location_wp, num_boxes, image_h, image_w, image_id, update_caption = img


        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_cls = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_attrs = np.zeros((self.region_len, 401), dtype=np.float32)
        image_location = np.zeros((self.region_len, self.num_locs), dtype=np.float32)

        # calculate the IOU here.
        overlaps = iou(image_location_wp, image_location_wp)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_cls[:num_boxes] = image_cls_wp
        image_attrs[:num_boxes] = attr_scores
        image_location[:num_boxes, :4] = image_location_wp
        obj_labels = obj_labels[:num_boxes]
        obj_confs = obj_confs[:num_boxes]
        attr_labels = attr_labels[:num_boxes]
        attr_confs = attr_confs[:num_boxes]

        if self.num_locs >= 5:
            image_location[:, -1] = (
                    (image_location[:, 3] - image_location[:, 1])
                    * (image_location[:, 2] - image_location[:, 0])
                    / (float(image_w) * float(image_h))
            )

        # Normalize the box locations (to 0 ~ 1)
        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        if self.num_locs > 5:
            image_location[:, 4] = image_location[:, 2] - image_location[:, 0]
            image_location[:, 5] = image_location[:, 3] - image_location[:, 1]

        tokens_sample_caption = None
        if self.tokenizer_name == "bert-base-uncased":
            tokens_caption = self.tokenizer.encode(update_caption)
            if sample_caption:
                tokens_sample_caption = self.tokenizer.encode(sample_caption)
        else:
            tokens_caption = self.tokenizer.encode(update_caption, add_special_tokens=False)
            if sample_caption:
                tokens_sample_caption = self.tokenizer.encode(sample_caption, add_special_tokens=False)

        cur_example = InputExample(
            image_feat=image_feature,
            image_cls=image_cls,
            obj_labels=obj_labels,
            obj_confs=obj_confs,
            attr_labels=attr_labels,
            attr_confs=attr_confs,
            image_attrs=image_attrs,
            caption=tokens_caption,
            sample_caption = tokens_sample_caption,
            order = neg_lang,
            is_next=label,
            image_loc=image_location,
            num_boxes=num_boxes,
            overlaps=overlaps,
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_cls,
            cur_features.obj_labels,
            cur_features.obj_confs,
            cur_features.attr_labels,
            cur_features.attr_confs,
            cur_features.image_attrs,
            cur_features.image_label,
            cur_features.image_mask,
            cur_features.masked_label,
            image_id,
        )
        return cur_tensors

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        """
        """
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        image_cls = example.image_cls
        num_boxes = int(example.num_boxes)
        overlaps = example.overlaps

        sample_tokens = None
        order = None
        sample_label =None
        lm_label_ids =None

        if example.sample_caption and example.order:
            sample_tokens = example.sample_caption
            order = example.order

            self._truncate_seq_pair(sample_tokens, max_seq_length/2 - 2)
            self._truncate_seq_pair(tokens, max_seq_length/2 - 2)
            # tokens_all, tokens_label_all = self.random_word(tokens+sample_tokens, tokenizer)
            # tokens = tokens_all[:length_token]
            # tokens_label = tokens_label_all[:length_token]
            # sample_tokens = tokens_all[length_token:]
            # sample_label = tokens_label_all[length_token:]
            tokens, tokens_label = self.random_word(tokens, tokenizer)
            sample_tokens, sample_label = self.random_word(sample_tokens, tokenizer)

        else:
            self._truncate_seq_pair(tokens, max_seq_length - 2)
            tokens, tokens_label = self.random_word(tokens, tokenizer)
            lm_label_ids = [-1] + tokens_label + [-1]

        image_feat, image_loc, image_label, masked_label = self.random_region(
            image_feat, image_loc, num_boxes, overlaps
        )

        # concatenate lm labels and account for CLS and SEP: [CLS] tokens [SEP]
        if self.tokenizer_name == "bert-base-uncased":
            if example.sample_caption and example.order:
                if order=='first':
                    tokens = tokenizer.add_special_tokens_single_sentence(sample_tokens, tokens)
                    lm_label_ids = [-1] + sample_label + [-1] + tokens_label + [-1]
                elif order=='second':
                    tokens = tokenizer.add_special_tokens_single_sentence(tokens, sample_tokens)
                    lm_label_ids = [-1] + tokens_label + [-1] + sample_label + [-1]
                else:
                    print('Error in convert_example_to_features bert-base-uncased tokenizer')
            else:
                tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        else:
            if example.sample_caption and example.order:
                if order=='first':
                    tokens = tokenizer.build_inputs_with_special_tokens(sample_tokens, tokens)
                    lm_label_ids = [-1] + sample_label + [-1] + [-1] + tokens_label + [-1]
                elif order=='second':
                    tokens = tokenizer.build_inputs_with_special_tokens(tokens, sample_tokens)
                    lm_label_ids = [-1] + tokens_label + [-1] + [-1] + sample_label + [-1]
                else:
                    print('Error in convert_example_to_features other tokenizer')
            else:
                tokens = tokenizer.build_inputs_with_special_tokens(tokens)
        segment_ids = [0] * len(tokens)

        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        image_mask = [1] * num_boxes

        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)


        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_cls=image_cls,
            obj_labels=example.obj_labels,
            obj_confs=example.obj_confs,
            attr_labels=example.attr_labels,
            attr_confs=example.attr_confs,
            image_attrs=example.image_attrs,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask=np.array(image_mask),
            masked_label = masked_label
        )
        return features

    def prepare_langauge_sampler(self, path):
        if path is None or path == "":
            return LanguageSamplingDefault(self.captions)
        elif os.path.exists(path):
            return LanguageSamplingGiven(path)
        else:
            assert False, "path should be `None` or exist:\n{}".format(path)

    @staticmethod
    def load_captions(path, split, langs):
        def load1(lang):
            with open(os.path.join(path, f"{lang}-{split}.json")) as f:
                return json.load(f)
        return {lang: load1(lang) for lang in langs}


    def random_cap(self, caption, image_id, neg_img=0.5, neg_lang1=0.25, neg_lang2=0.25):
        neg_lang = None
        sample_caption = None
        img = None
        if not self.visualization and self.objective != 2 and random.random() > 0.5:
            # Pick a random caption from a random language.
            random_sample_value = random.uniform(0, 1)
            lang = random.choice(self.langs)
            if random_sample_value < neg_img:
                # img = random.choice(list(self.ds_preprocess))
                caption = random.choice(list(self.captions[lang].values()))
                lang = random.choice(self.langs)
                sample_caption = random.choice(list(self.captions[lang].values()))
                neg_lang = "second"
            elif random_sample_value >= neg_img and random_sample_value < neg_img + neg_lang1:
                neg_lang = "first"
                sample_caption = random.choice(list(self.captions[lang].values()))
            else:
                neg_lang = "second"
                sample_caption = random.choice(list(self.captions[lang].values()))
            label = 1
        else:
            lang = self.sample_language(image_id)
            sample_caption = self.captions[lang][image_id]
            if random.random() > 0.5:
                neg_lang = "first"
            else:
                neg_lang = "second"
            label = 0
        return caption, sample_caption, neg_lang, label, image_id, img

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
    def __init__(self, path):
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

    def __call__(self, image_id):
        p = self.p_lang_given_sent[image_id]
        return random.choices(self.langs, weights=p)[0]



class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
            self,
            image_feat=None,
            image_cls=None,
            obj_labels=None,
            obj_confs=None,
            attr_labels=None,
            attr_confs=None,
            image_attrs=None,
            caption=None,
            order= None,
            is_next=None,
            lm_labels=None,
            image_loc=None,
            num_boxes=None,
            overlaps=None,
            sample_caption =None
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_cls = image_cls
        self.obj_labels = obj_labels  # (label, conf)
        self.obj_confs = obj_confs
        self.attr_labels = attr_labels  # (label, conf)
        self.attr_confs = attr_confs
        self.image_attrs = image_attrs
        self.num_boxes = num_boxes
        self.overlaps = overlaps
        self.sample_caption = sample_caption
        self.order = order