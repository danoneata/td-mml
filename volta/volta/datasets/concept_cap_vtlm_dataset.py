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
from torch.utils.data import Dataset

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

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids=None,
            input_mask=None,
            segment_ids=None,
            position_ids=None,
            lm_label_ids=None,
            image_feat=None,
            image_cls=None,
            obj_labels=None,
            obj_confs=None,
            attr_labels=None,
            attr_confs=None,
            image_attrs=None,
            image_loc=None,
            image_label=None,
            image_mask=None,
            masked_label=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_cls = image_cls
        self.obj_labels = obj_labels
        self.obj_confs = obj_confs
        self.attr_labels = attr_labels
        self.attr_confs = attr_confs
        self.image_attrs = image_attrs
        self.image_mask = image_mask
        self.masked_label = masked_label

class ConceptCapVTLM_LoaderTrain(Dataset):
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
        self.lmdb_file = copy(ds)
        ds_preprocess = copy(ds)

        self.num_dataset = len(ds)
        ds = td.LocallyShuffleData(ds, cache)

        preprocess_function = BertPreprocessBatch_vtlm(
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

    def __iter__(self):
        for batch in self.ds.get_data():

            input_ids, input_mask, segment_ids,position_ids, lm_label_ids, image_feat, image_loc, \
            image_cls, obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, image_label, image_mask, masked_label, image_id = batch

            batch_size = input_ids.shape[0]

            if self.add_global_imgfeat == "first":
                sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
                sum_count[sum_count == 0] = 1
                g_image_feat = np.sum(image_feat, axis=1) / sum_count
                image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
                image_feat = np.array(image_feat, dtype=np.float32)

                g_loc = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
                g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)

                image_loc = np.array(image_loc, dtype=np.float32)
                g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            elif self.add_global_imgfeat == "last":
                sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
                sum_count[sum_count == 0] = 1
                g_image_feat = np.sum(image_feat, axis=1) / sum_count
                image_feat = np.concatenate([image_feat, np.expand_dims(g_image_feat, axis=1)], axis=1)
                image_feat = np.array(image_feat, dtype=np.float32)

                g_loc = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
                g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_loc = np.concatenate([image_loc, np.expand_dims(g_image_loc, axis=1)], axis=1)

                image_loc = np.array(image_loc, dtype=np.float32)
                g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_mask = np.concatenate([image_mask, g_image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                position_ids,
                lm_label_ids,
                image_feat,
                image_loc,
                image_cls,
                obj_labels,
                obj_confs,
                attr_labels,
                attr_confs,
                image_attrs,
                image_label,
                image_mask,
            )
            yield tuple([torch.tensor(data) for data in batch] + [image_id])

    def __len__(self):
        return self.ds.size()

class ConceptCapVTLM_LoaderVal(Dataset):
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

class BertPreprocessBatch_vtlm(BertPreprocessBatch):
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

        caption_en, caption_x, neg_lang = self.create_mlm_io(caption, image_id=image_id,
                                                               neg_lang1=0.5, neg_lang2=0.5)


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

        tokens_caption_x = None
        if self.tokenizer_name == "bert-base-uncased":
            tokens_caption_en = self.tokenizer.encode(caption_en)
            if caption_x:
                tokens_caption_x = self.tokenizer.encode(caption_x)
        else:
            tokens_caption_en = self.tokenizer.encode(caption_en, add_special_tokens=False)
            if caption_x:
                tokens_caption_x = self.tokenizer.encode(caption_x, add_special_tokens=False)

        cur_example = InputExample(
            image_feat=image_feature,
            image_cls=image_cls,
            obj_labels=obj_labels,
            obj_confs=obj_confs,
            attr_labels=attr_labels,
            attr_confs=attr_confs,
            image_attrs=image_attrs,
            caption_en= tokens_caption_en,
            caption_x = tokens_caption_x,
            order = neg_lang,
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
            cur_features.position_ids,
            cur_features.lm_label_ids,
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

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length,sequence_a_segment_id=0, sequence_b_segment_id=1,cls_token_segment_id=2, sep_token_extra=True):
        """
        """
        image_feat = example.image_feat
        tokens_en = example.caption_en
        image_loc = example.image_loc
        image_cls = example.image_cls
        num_boxes = int(example.num_boxes)
        overlaps = example.overlaps

        sample_tokens = None
        order = None
        sample_label =None
        lm_label_ids =None

        tokens_en_label=None
        tokens_x_label =None
        tokens_x=None

        if example.caption_x and example.order:
            tokens_x = example.caption_x
            order = example.order

            special_tokens_count = 4 if sep_token_extra else 3
            self._truncate_seq_pairs(tokens_en, tokens_x, max_seq_length - special_tokens_count)

            tokens_en, tokens_en_label = self.random_word(tokens_en, tokenizer)
            tokens_x, tokens_x_label = self.random_word(tokens_x, tokenizer)

        else:
            self._truncate_seq_pair(tokens_en, max_seq_length - 2)
            tokens_en, tokens_label = self.random_word(tokens_en, tokenizer)
            lm_label_ids = [-1] + tokens_label + [-1]

        # image_feat, image_loc, image_label, masked_label = self.random_region(
        #     image_feat, image_loc, num_boxes, overlaps
        # )
        image_label= [-1] * num_boxes
        masked_label = np.zeros((image_feat.shape[0]))

        segment_ids = None
        tokens=None
        # concatenate lm labels and account for CLS and SEP: [CLS] tokens [SEP]
        if self.tokenizer_name == "bert-base-uncased":
            if example.caption_x and example.order:
                if order=='first':
                    tokens = tokenizer.add_special_tokens_single_sentence(tokens_en, tokens_x)
                    lm_label_ids = [-1] + tokens_en_label + [-1] + tokens_x_label + [-1]
                    segment_ids = [0] + [0] * len(tokens_en) + [1] + [1] * len(tokens_x) + [1]
                elif order=='second':
                    tokens = tokenizer.add_special_tokens_single_sentence(tokens_x, tokens_en)
                    lm_label_ids = [-1] + tokens_x_label + [-1] + tokens_en_label + [-1]
                    segment_ids = [0] + [0] * len(tokens_x) + [1] + [1] * len(tokens_en) + [1]
                else:
                    print('Error in convert_example_to_features bert-base-uncased tokenizer')
            else:
                tokens = tokenizer.add_special_tokens_single_sentence(tokens_en)
                segment_ids = [0] * len(tokens)
        else:
            if example.caption_x and example.order:
                if order=='first':
                    tokens = tokenizer.build_inputs_with_special_tokens(tokens_en, tokens_x)
                    lm_label_ids = [-1] + tokens_en_label + [-1] + [-1] + tokens_x_label + [-1]
                    segment_ids = [0] + [0] * len(tokens_en) + [0] + [1] + [1] * len(tokens_x) + [1]
                elif order=='second':
                    tokens = tokenizer.build_inputs_with_special_tokens(tokens_x, tokens_en)
                    lm_label_ids = [-1] + tokens_x_label + [-1] + [-1] + tokens_en_label + [-1]
                    segment_ids = [0]+ [0] * len(tokens_x)+ [0] + [1]+ [1] * len(tokens_en) + [1]
                else:
                    print('Error in convert_example_to_features other tokenizer')
            else:
                tokens = tokenizer.build_inputs_with_special_tokens(tokens_en)
                segment_ids = [0] * len(tokens)

        # segment_ids = [0] * len(tokens)
        input_ids = tokens
        position_ids = [0] * len(tokens)

        #create position_ids, specific built for XLMR if we set position_ids instead of segment_ids
        # if sequence_b_segment_id in segment_ids:
        #     position_ids = [0] * len(tokens)
        # else:
        #     position_ids = []
        #     position_id = 0
        #     for id in range(len(tokens_en_label)+2):
        #         position_id += 1
        #         position_ids.append(position_id)
        #     position_id = 0
        #     for id in range(len(tokens_x_label)+2):
        #         position_id += 1
        #         position_ids.append(position_id)

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
            position_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length
        assert len(position_ids) == max_seq_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            position_ids=np.array(position_ids),
            lm_label_ids=np.array(lm_label_ids),
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

    def create_mlm_io(self, caption, image_id, neg_lang1=0.5, neg_lang2=0.5):
        lang = self.sample_language(image_id)
        sample_caption = self.captions[lang][image_id]
        if random.random() > neg_lang1:
            neg_lang = "first"
        else:
            neg_lang = "second"
        return caption, sample_caption, neg_lang

    def _truncate_seq_pairs(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

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
            caption_en=None,
            order= None,
            lm_labels=None,
            image_loc=None,
            num_boxes=None,
            overlaps=None,
            caption_x =None
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
        self.caption_x = caption_x
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
        self.caption_en = caption_en
        self.order = order