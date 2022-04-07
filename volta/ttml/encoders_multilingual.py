from torch import nn
from volta.encoders import (
    BertForVLPreTraining,
    BertModel,
    BertPreTrainingHeads
)
from volta.losses import pre_vis_criterions, pre_vis_targets
import torch
from torch import nn
from torch.nn import functional as F

class BertForVLPreTraining_Multilingual(BertForVLPreTraining):
    def __init__(self, config):
        super(BertForVLPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.visual_target_weights = config.visual_target_weights
        print("model's visual targets are ", [ix for ix, w in config.visual_target_weights.items() if w > 0])

        self.add_global_imgfeat = int(config.add_global_imgfeat is not None)

        self.tie_weights()
        self.itm_output = nn.Linear(config.pooler_size, 2)

    def forward(self, batch, task):
        input_ids, image_feat, image_loc, segment_ids,input_mask, image_mask, lm_label_ids, image_label,\
        image_cls, obj_labels, obj_confs, attr_labels,attr_confs, image_attrs, is_match = batch
        if task.startswith('mlm'):
            return self.forward_mlm(input_ids, image_feat, image_loc, segment_ids,
                                                                  input_mask, image_mask, lm_label_ids, image_label,
                                                                  image_cls, obj_labels, obj_confs, attr_labels,
                                                                  attr_confs, image_attrs, is_match)
        elif task.startswith('itm'):
            targets = is_match
            return self.forward_itm(input_ids, image_feat, image_loc, segment_ids,
                                                                  input_mask, image_mask, lm_label_ids, image_label,
                                                                  image_cls, obj_labels, obj_confs, attr_labels,
                                                                  attr_confs, image_attrs, targets)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self,input_ids, image_feat, image_loc,token_type_ids=None, attention_mask=None,
                    image_attention_mask=None,masked_lm_labels=None, image_label=None, image_cls=None,
                    obj_labels=None, obj_confs=None, attr_labels=None, attr_confs=None, image_attrs=None,
                    next_sentence_label=None,output_all_encoded_layers=False, output_all_attention_masks=False
                    ):
        # in this model, we first embed the images.
        encoded_layers_t, encoded_layers_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )
        if output_all_encoded_layers:
            sequence_output_t = encoded_layers_t[-1]
            sequence_output_v = encoded_layers_v[-1]
        else:
            sequence_output_t = encoded_layers_t
            sequence_output_v = encoded_layers_v

        prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, pooled_output = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        # Vision loss
        img_loss = 0
        for ix, weight in self.visual_target_weights.items():
            if self.config.add_global_imgfeat == "last":
                prediction_scores_v = prediction_scores_v_dict[ix][:, :-1]
            else:
                prediction_scores_v = prediction_scores_v_dict[ix][:, self.add_global_imgfeat:]
            img_loss += pre_vis_criterions[ix](prediction_scores_v, weight, image_label, image_cls, image_feat,
                                               obj_labels, obj_confs, attr_labels, attr_confs)

        masked_img_loss = img_loss > 0 if type(img_loss) == int else img_loss.cpu().item() > 0
        if masked_img_loss:
            img_loss = img_loss.unsqueeze(0)
        else:
            img_loss = torch.zeros(1).to(input_ids.device)

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            ).unsqueeze(0)
        else:
            masked_lm_loss = torch.zeros(1).to(input_ids.device)

        if (seq_relationship_score is not None) and (next_sentence_label is not None):
            next_sentence_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            ).unsqueeze(0)
        else:
            next_sentence_loss = torch.zeros(1).to(input_ids.device)

        if masked_img_loss or masked_lm_loss or next_sentence_loss:
            if output_all_encoded_layers:
                return masked_lm_loss, img_loss, next_sentence_loss, encoded_layers_t, encoded_layers_v
            return masked_lm_loss, img_loss, next_sentence_loss
        else:
            if output_all_encoded_layers:
                return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, all_attention_mask, \
                       pooled_output, encoded_layers_t, encoded_layers_v
            return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, all_attention_mask, pooled_output

    def forward_itm(self,input_ids, image_feat, image_loc,token_type_ids=None, attention_mask=None,
                    image_attention_mask=None,masked_lm_labels=None, image_label=None, image_cls=None,
                    obj_labels=None, obj_confs=None, attr_labels=None, attr_confs=None, image_attrs=None,
                    next_sentence_label=None,output_all_encoded_layers=False, output_all_attention_masks=False
                    ):
        # in this model, we first embed the images.
        encoded_layers_t, encoded_layers_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )
        if output_all_encoded_layers:
            sequence_output_t = encoded_layers_t[-1]
            sequence_output_v = encoded_layers_v[-1]
        else:
            sequence_output_t = encoded_layers_t
            sequence_output_v = encoded_layers_v

        prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, pooled_output = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        # Vision loss
        img_loss = 0
        for ix, weight in self.visual_target_weights.items():
            if self.config.add_global_imgfeat == "last":
                prediction_scores_v = prediction_scores_v_dict[ix][:, :-1]
            else:
                prediction_scores_v = prediction_scores_v_dict[ix][:, self.add_global_imgfeat:]
            img_loss += pre_vis_criterions[ix](prediction_scores_v, weight, image_label, image_cls, image_feat,
                                               obj_labels, obj_confs, attr_labels, attr_confs)

        masked_img_loss = img_loss > 0 if type(img_loss) == int else img_loss.cpu().item() > 0
        if masked_img_loss:
            img_loss = img_loss.unsqueeze(0)
        else:
            img_loss = torch.zeros(1).to(input_ids.device)

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            ).unsqueeze(0)
        else:
            masked_lm_loss = torch.zeros(1).to(input_ids.device)

        if (seq_relationship_score is not None) and (next_sentence_label is not None):
            next_sentence_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            ).unsqueeze(0)
        else:
            next_sentence_loss = torch.zeros(1).to(input_ids.device)

        if masked_img_loss or masked_lm_loss or next_sentence_loss:
            if output_all_encoded_layers:
                return masked_lm_loss, img_loss, next_sentence_loss, encoded_layers_t, encoded_layers_v
            return masked_lm_loss, img_loss, next_sentence_loss
        else:
            if output_all_encoded_layers:
                return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, all_attention_mask, \
                       pooled_output, encoded_layers_t, encoded_layers_v
            return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, all_attention_mask, pooled_output

