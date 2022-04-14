#!/bin/bash
#SBATCH --job-name=lg-random-pretrain-cc
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --mem=10GB
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=10-12:00:00
#SBATCH --output="pretrain-cc.multilingual.random-lg.log"


DATA=/science/image/nlp-datasets/emanuele/data/conceptual_captions
DIR=/science/image/nlp-datasets/tt-mml
FEATS=$DATA/resnet101_faster_rcnn_genome_imgfeats/volta
ANNOS=$DIR/data/conceptual_captions/annotations/m2m-100-lg-seed-1337
MODEL_CONFIG=ctrl_xuniter_base

LANG_SAMPLING=""
STRATEGY=random_lg
TRAIN_BATCH_SIZE=256
WARMUP_Proportion=0.05

TETASK=conceptual_captions-${STRATEGY}
OUTPUT_DIR=$DIR/checkpoints/iglue/pretrain/ctrl_ttmml_${STRATEGY}/${MODEL_CONFIG}/train_batch_size_${TRAIN_BATCH_SIZE}/${TETASK}
LOGGING_DIR=$$DIR/logs/pretrain/ctrl_ttmml_${STRATEGY}/${MODEL_CONFIG}/train_batch_size_${TRAIN_BATCH_SIZE}/${TETASK}

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../volta

python train_concap_multilingual.py \
  --bert_model xlm-roberta-base \
  --from_pretrained xlm-roberta-base \
  --config_file config/${MODEL_CONFIG}.json \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps 4 \
  --max_seq_length 66 \
  --learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion $WARMUP_Proportion --clip_grad_norm 5.0 \
  --objective 1 \
  --annotations_path $ANNOS \
  --features_path $FEATS \
  --output_dir $OUTPUT_DIR \
  --logdir $LOGGING_DIR \
  --num_train_epochs 10 \
  --langs_sampling_path $LANG_SAMPLING \
  --save_every_n_steps 10000
  # --num_workers 0  # For debugging purposes

deactivate