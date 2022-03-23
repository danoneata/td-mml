
import lmdb
import msgpack
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import os
import json
import random


class ConcatDatasetWithLens(ConcatDataset):
    """ A thin wrapper on pytorch concat dataset for lens batching """
    def __init__(self, datasets):
        super().__init__(datasets)
        self.lens = [l for dset in datasets for l in dset.lens]

    def __getattr__(self, name):
        return self._run_method_on_all_dsets(name)

    def _run_method_on_all_dsets(self, name):
        def run_all(*args, **kwargs):
            return [dset.__getattribute__(name)(*args, **kwargs)
                    for dset in self.datasets]
        return run_all

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
            img_id=None,
            lm_labels=None,
            image_loc=None,
            num_boxes=None,
            overlaps=None,
            is_next=None,
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
        self.img_id = img_id
        self.is_next = is_next
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



class DetectFeatLmdb(object):
    def __init__(self, img_dir, args, config):
        self.img_dir = img_dir
        self.args = args
        self.config = config
        self.region_len = 36
        self.num_locs = config.num_locs


        self.FIELDNAMES = ["features", "cls_prob", "objects_id", "objects_conf", "attrs_id",
                           "attrs_conf", "attrs", "boxes", "num_boxes", "img_h",
                           "img_w", "img_id", "caption"]
        if 'train' in img_dir.split('/')[-1]:
            db_name = 'training_feat_all.lmdb'
            self.split = 'train'
        else:
            db_name = 'validation_feat_all.lmdb'
            self.split = 'valid'

        self.env = lmdb.open(f'{img_dir}/{db_name}', subdir=False,
                                 readonly=True, create=False,
                                 readahead=False)
        self.txn = self.env.begin(write=False)

        self.keys = list(self.txn.cursor().iternext(values=False))
        self.num_dataset = len(self.keys)

    def __del__(self):
        self.env.close()

    def get_dump(self, file_name):
        dump = self.txn.get(file_name.encode('utf-8'))
        img_dump = msgpack.loads(dump, raw=False)
        img_dump = dict(zip(self.FIELDNAMES, img_dump))
        preprocessed_img_dump = self.preprocess(img_dump)
        return preprocessed_img_dump

    def __getitem__(self, file_name):
        dump = self.txn.get(file_name.encode('utf-8'))
        img_dump = msgpack.loads(dump, raw=False)
        img_dump = dict(zip(self.FIELDNAMES, img_dump))
        preprocessed_img_dump = self.preprocess(img_dump)
        return preprocessed_img_dump

    def __contains__(self, file_name):
        return self.txn.get(file_name.encode('utf-8')) is not None

    def preprocess(self, img_dump):

        self.FIELDNAMES = ["features", "cls_prob", "objects_id", "objects_conf", "attrs_id",
                           "attrs_conf", "attrs", "boxes", "num_boxes", "img_h",
                           "img_w", "img_id", "caption"]

        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_cls = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_attrs = np.zeros((self.region_len, 401), dtype=np.float32)
        image_location = np.zeros((self.region_len, self.num_locs), dtype=np.float32)

        # calculate the IOU here.
        overlaps = iou(img_dump["box"], img_dump["box"])
        num_boxes = int(img_dump['num_boxes'])
        image_feature[:num_boxes] = img_dump['features']
        image_cls[:num_boxes] = img_dump['cls_prob']
        image_attrs[:num_boxes] = img_dump['attrs']
        image_location[:num_boxes, :4] = img_dump["box"]
        obj_labels = img_dump['objects_id'][:num_boxes]
        obj_confs = img_dump['objects_conf'][:num_boxes]
        attr_labels = img_dump['attrs_id'][:num_boxes]
        attr_confs = img_dump['attrs_conf'][:num_boxes]

        if self.num_locs >= 5:
            image_location[:, -1] = (
                    (image_location[:, 3] - image_location[:, 1])
                    * (image_location[:, 2] - image_location[:, 0])
                    / (float(img_dump['img_w']) * float(img_dump['img_h']))
            )
        # Normalize the box locations (to 0 ~ 1)
        image_location[:, 0] = image_location[:, 0] / float(img_dump['img_w'])
        image_location[:, 1] = image_location[:, 1] / float(img_dump['img_h'])
        image_location[:, 2] = image_location[:, 2] / float(img_dump['img_w'])
        image_location[:, 3] = image_location[:, 3] / float(img_dump['img_h'])

        if self.num_locs > 5:
            image_location[:, 4] = image_location[:, 2] - image_location[:, 0]
            image_location[:, 5] = image_location[:, 3] - image_location[:, 1]

        # We don't do the random_cap and convert_example_to_features (refering concept_cap_dataset.py) here, but do it
        # in the detailed dataset preprocess function, et.al collate_fn in build_mlm_dataset
        cur_example = InputExample(
            image_feat=image_feature,
            image_cls=image_cls,
            obj_labels=obj_labels,
            obj_confs=obj_confs,
            attr_labels=attr_labels,
            attr_confs=attr_confs,
            image_attrs=image_attrs,
            caption=img_dump['caption'],
            img_id=img_dump['img_id'],
            image_loc=image_location,
            num_boxes=num_boxes,
            overlaps=overlaps,
        )
        return cur_example


# class ImageLmdbGroup(object):
#     def __init__(self, args, config, is_train):
#         self.path2imgdb = {}
#         self.config = config
#         self.args = args
#
#     def __getitem__(self, path):
#         img_db = self.path2imgdb.get(path, None)
#         #print(path)
#         if img_db is None:
#             img_db = DetectFeatLmdb(path, self.args, self.config)
#         return img_db


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids=None,
            input_mask=None,
            segment_ids=None,
            is_next=None,
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
        self.is_next = is_next
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

class DetectFeatTxtTokDataset(Dataset):
    def __init__(self, img_db,is_train, args, config,tokenizer,visualization =False,task_name="mlm"):
        assert isinstance(img_db, DetectFeatLmdb)
        self.img_db = img_db
        self.ids = img_db.keys
        self.num_dataset = len(self.ids)
        self.visualization =False
        self.seq_len = args.max_seq_length
        self.tokenizer = tokenizer
        self.tokenizer_name = args.bert_model
        self.objective = args.objective
        self.region_len = 36

        self.add_global_imgfeat = config.add_global_imgfeat
        self.num_locs = config.num_locs


        if is_train:
            split = 'train'
        else:
            split = 'valid'
        self.captions = self.load_captions(args.annotations_path, split, args.langs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.img_db[id_]
        update_example = self.Preprocess(example)
        return update_example

    @staticmethod
    def load_captions(path, split, langs):
        def load1(lang):
            with open(os.path.join(path, f"{lang}-{split}.json")) as f:
                return json.load(f)
        return {lang: load1(lang) for lang in langs}

    def Preprocess(self, cur_example):
        caption, label = self.random_cap(cur_example.caption,image_id=cur_example.img_id)
        if self.tokenizer_name == "bert-base-uncased":
            tokens_caption = self.tokenizer.encode(caption)
        else:
            tokens_caption = self.tokenizer.encode(caption, add_special_tokens=False)
        cur_example.caption = tokens_caption
        cur_example.label = label

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
            cur_example.img_id,
            self.add_global_imgfeat,
            self.num_locs
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

        self._truncate_seq_pair(tokens, max_seq_length - 2)

        tokens, tokens_label = self.random_word(tokens, tokenizer)
        image_feat, image_loc, image_label, masked_label = self.random_region(
            image_feat, image_loc, num_boxes, overlaps
        )

        # concatenate lm labels and account for CLS and SEP: [CLS] tokens [SEP]
        lm_label_ids = [-1] + tokens_label + [-1]
        if self.tokenizer_name == "bert-base-uncased":
            tokens = tokenizer.add_special_tokens_single_sentence(tokens)
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
            masked_label=masked_label,
        )
        return features

    def random_cap(self, caption, image_id):
        """Replaces the current caption with probabilty 50%."""
        if self.visualization:
            return caption, 0

        if random.random() > 0.5:
            caption = self.get_random_caption()
            label = 1
        else:
            label = 0

        return caption, label

    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        lang = 'en'
        caption = random.choice(list(self.captions[lang].values()))

        return caption

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break
            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes, overlaps):
        """
        """
        output_label = []
        masked_label = np.zeros((image_feat.shape[0]))

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 90% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # mask the overlap regions into zeros
                masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label, masked_label

