from toolz.sandbox import unzip
import numpy as np
import torch
import random
import os
from .data import (DetectFeatTxtTokDataset)

class MlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, img_db,is_train, args, config,tokenizer):
        super().__init__(img_db,is_train, args, config,tokenizer)
        self.num_dataset = super().num_dataset

    def __getitem__(self, i):
        example = super().__getitem__(i)
        example_tensors = super().Preprocess(example)
        return example_tensors

class MlmDataset_Multilingual(DetectFeatTxtTokDataset):
    def __init__(self, img_db,is_train, args, config,tokenizer, task_name="mlm"):
        super().__init__(img_db,is_train, args, config,tokenizer,task_name="mlm")
        self.captions = super().captions
        self.num_dataset = super().num_dataset
        self.sample_language = self.prepare_langauge_sampler(args.langs_sampling_path)

    def __getitem__(self, i):
        example = super().__getitem__(i)
        example_tensors = super().Preprocess(example)
        return example_tensors

    def random_cap(self, caption, image_id):
        if not self.visualization and random.random() > 0.5:
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



    def prepare_langauge_sampler(self, path):

        if path is None:
            return LanguageSamplingDefault(self.captions)
        elif os.path.exists(path):
            return LanguageSamplingGiven(path)
        else:
            assert False, "path should be `None` or exist:\n{}".format(path)

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


def calculate(input):
    add_global_imgfeat, num_locs, masked_label, image_feat, batch_size, image_mask, image_loc= input

    if add_global_imgfeat == "first":
        sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
        sum_count[sum_count == 0] = 1
        g_image_feat = np.sum(image_feat, axis=1) / sum_count
        image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
        image_feat = np.array(image_feat, dtype=np.float32)

        g_loc = [0, 0, 1, 1] + [1] * (num_locs - 4)
        g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
        image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)

        image_loc = np.array(image_loc, dtype=np.float32)
        g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
        image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

    elif add_global_imgfeat == "last":
        sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
        sum_count[sum_count == 0] = 1
        g_image_feat = np.sum(image_feat, axis=1) / sum_count
        image_feat = np.concatenate([image_feat, np.expand_dims(g_image_feat, axis=1)], axis=1)
        image_feat = np.array(image_feat, dtype=np.float32)

        g_loc = [0, 0, 1, 1] + [1] * (num_locs - 4)
        g_image_loc = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
        image_loc = np.concatenate([image_loc, np.expand_dims(g_image_loc, axis=1)], axis=1)

        image_loc = np.array(image_loc, dtype=np.float32)
        g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
        image_mask = np.concatenate([image_mask, g_image_mask], axis=1)

    return image_mask, image_loc

def mlm_multilingual_collate(inputs):
    input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
    image_loc, image_cls, obj_labels, obj_confs, attr_labels, attr_confs, \
    image_attrs, image_label, image_mask, masked_label, image_id, \
    add_global_imgfeat, num_locs= inputs

    batch_size = input_ids.shape[0]

    calculate_input = (add_global_imgfeat, num_locs, masked_label, image_feat,batch_size,image_mask,image_loc)

    image_mask, image_loc = calculate(calculate_input)

    batch = (
            input_ids,
            input_mask,
            segment_ids,
            lm_label_ids,
            is_next,
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
    return tuple([torch.tensor(data) for data in batch] + [image_id])


def mlm_collate(inputs):
    # (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc,
    #  image_cls, obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, image_label, image_mask, masked_label, image_id,
    #  add_global_imgfeat, num_locs
    #  ) = map(list, unzip(inputs))

    input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
    image_loc, image_cls, obj_labels, obj_confs, attr_labels, attr_confs, \
    image_attrs, image_label, image_mask, masked_label, image_id, \
    add_global_imgfeat, num_locs= inputs

    batch_size = input_ids.shape[0]

    calculate_input = (add_global_imgfeat, num_locs, masked_label, image_feat,batch_size,image_mask,image_loc)

    image_mask, image_loc = calculate(calculate_input)

    batch = (
            input_ids,
            input_mask,
            segment_ids,
            lm_label_ids,
            is_next,
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
    return tuple([torch.tensor(data) for data in batch] + [image_id])

