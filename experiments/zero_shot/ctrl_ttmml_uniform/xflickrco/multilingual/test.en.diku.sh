#!/bin/bash
#SBATCH --job-name=en-xflickrco
#SBATCH --ntasks=1
#SBATCH -p gpu --gres=gpu:titanrtx:1 --mem=40GB
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00

TASK=8
LANG=en
MODEL=ctrl_xuniter_simplett
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=iglue_test_tasks_boxes36.dtu
TRTASK=RetrievalFlickr30k
TETASK=RetrievalxFlickrCO_${LANG}

TEXT_PATH=/home/mwp141/Prepared/iglue/volta/data/trans_cc/m2m-100-lg/${LANG}-valid-1000.json
#FEAT_PATH=/home/mwp141/Working/volta/data/xflickro/xflickrco-test_boxes36.lmdb
FEAT_PATH=/home/mwp141/Working/volta/data/conceptual_captions/resnet101_faster_rcnn_genome_imgfeats/volta
PRETRAINED=/home/mwp141/Prepared/iglue/volta/checkpoints/conceptual_captions/ctrl_xuniter_tt_simple/ctrl_xuniter_base/pytorch_model_9.bin
OUTPUT_DIR=/home/mwp141/Prepared/iglue/experiments/zero_shot/ctrl_xuniter/xflickrco/${MODEL}/$TETASK

ID2Key_DIR=/home/mwp141/Prepared/iglue/volta/data/trans_cc

cd ../../../../../volta
python eval_retrieval.py \
  --bert_model xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} --num_val_workers 0 \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test_${LANG} --batch_size 1 \
  --caps_per_image 1 --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH} \
  --id2key_dir ${ID2Key_DIR}\
  --output_dir ${OUTPUT_DIR} \
  --zero_shot
