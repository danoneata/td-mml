from volta.datasets.retrieval_dataset import RetrievalDatasetVal
from ._image_features_reader import ImageFeaturesH5Reader_cc
import os
import sys
import json
import random
import logging
import jsonlines
import _pickle as cPickle

import base64
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import msgpack_numpy
msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
Valid2key_name= "validation_id2key.json"

def _load_annotations(annotations_jsonpath, id2key_path):
    caption_entries = []
    image_entries = {}
    with open(annotations_jsonpath, 'r', encoding='utf-8')as fp:
        json_data = json.load(fp)

    id2key_path = os.path.join(id2key_path,Valid2key_name)
    # print('id2key_path', id2key_path)
    with open(id2key_path, 'r', encoding='utf-8')as fp:
        id2key = json.load(fp)
    # print("id2key", id2key)
    for image_id, value in json_data.items():
        key = id2key[image_id]
        caption_entries.append({"caption": value, "image_id": key})
        image_entries[key] = 1
    image_entries = [*image_entries]
    return image_entries, caption_entries


class RetrievalDatasetVal_Multilingual(RetrievalDatasetVal):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader_cc,
        gt_image_features_reader: ImageFeaturesH5Reader_cc,
        tokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 36,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
        num_subiters=2,
        id2key_dir=None
    ):
        self._image_entries, self._caption_entries = _load_annotations(annotations_jsonpath, id2key_dir)
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat
        self.num_labels = 1

        self.num_subiters = num_subiters
        self.num_images = len(self._image_entries)
        self.num_entries = len(self._caption_entries)
        self.max_num_images = self.num_images // self.num_subiters + int(self.num_images % self.num_subiters > 0)

        os.makedirs(os.path.join("/".join(annotations_jsonpath.split("/")[:-1]), "cache"), exist_ok=True)
        cache_path = os.path.join(
            "/".join(annotations_jsonpath.split("/")[:-1]),
            "cache",
            task
            + "_"
            + split
            + "_"
            + bert_model.split("/")[-1]
            + "_"
            + str(max_seq_length)
            + ".pkl",
        )
        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._caption_entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % cache_path)
            self._caption_entries = cPickle.load(open(cache_path, "rb"))

        self.features_all = np.zeros((len(self._image_entries), self._max_region_num, 2048))
        self.spatials_all = np.zeros((len(self._image_entries), self._max_region_num, self._num_locs))
        self.image_mask_all = np.zeros((len(self._image_entries), self._max_region_num))

        # print("self._image_features_reader.keys()",self._image_features_reader.keys())
        for i, image_id in enumerate(self._image_entries):
            features, num_boxes, boxes, _ = self._image_features_reader[image_id]

            mix_num_boxes = min(int(num_boxes), self._max_region_num)
            mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
            mix_features_pad = np.zeros((self._max_region_num, 2048))

            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < self._max_region_num:
                image_mask.append(0)

            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            self.features_all[i] = mix_features_pad
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = mix_boxes_pad

            sys.stdout.write("%d/%d\r" % (i, len(self._image_entries)))
            sys.stdout.flush()

        self.features_all = torch.Tensor(self.features_all).float()
        self.image_mask_all = torch.Tensor(self.image_mask_all).long()
        self.spatials_all = torch.Tensor(self.spatials_all).float()
