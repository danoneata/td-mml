#!/bin/bash
#SBATCH --job-name=20lg-random-pretrain-cc
#SBATCH --ntasks=1 --cpus-per-task=20 --mem=60GB
#SBATCH -p gpu --gres=gpu:titanrtx:2
#SBATCH --time=3-12:30:00
#SBATCH --output="pretrain-cc.multilingual.random-lg.ptr-20langs.epoch20.log"

LANGS=20
LANGS_SET=ar-bg-bn-da-de-el-en-es-et-fr-id-ja-ko-pt-ru-sw-ta-tr-vi-zh
DATA=/science/image/nlp-datasets/emanuele/data/conceptual_captions
DIR=/science/image/nlp-datasets/tt-mml
FEATS=$DATA/resnet101_faster_rcnn_genome_imgfeats/volta
ANNOS=$DIR/data/conceptual_captions/annotations/langs_${LANGS}/m2m-100-lg-filtered
MODEL_CONFIG=ctrl_xuniter_base

STRATEGY=random_lg
TRAIN_BATCH_SIZE=256
WARMUP_Proportion=0.05

echo "train_batch_size":$TRAIN_BATCH_SIZE
echo "strategy":$STRATEGY
echo "epochs=20"

TETASK=conceptual_captions-${STRATEGY}-epoch20
OUTPUT_DIR=$DIR/checkpoints/iglue/pretrain/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${MODEL_CONFIG}/train_batch_size_${TRAIN_BATCH_SIZE}/${TETASK}
LOGGING_DIR=$DIR/logs/pretrain/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${MODEL_CONFIG}/train_batch_size_${TRAIN_BATCH_SIZE}/${TETASK}

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../volta

python train_concap_multilingual.py \
  --bert_model xlm-roberta-base \
  --from_pretrained xlm-roberta-base \
  --config_file config/${MODEL_CONFIG}.json \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps 4 \
  --max_seq_length 40 \
  --learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion $WARMUP_Proportion --clip_grad_norm 5.0 \
  --objective 1 \
  --annotations_path $ANNOS \
  --features_path $FEATS \
  --output_dir $OUTPUT_DIR \
  --logdir $LOGGING_DIR \
  --num_train_epochs 20 \
  --langs_sampling_path "" \
  --save_every_n_steps 10000 \
  --langs $LANGS_SET \
  --num_workers 5
  # --num_workers 0  # For debugging purposes

deactivate