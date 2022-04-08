#!/bin/bash
#SBATCH --job-name=lgsample-a-pretrain-cc
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --mem=10GB
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=10-12:00:00
#SBATCH --output="pretrain.train_concap_multilingual.sample-a-mlm-100-lg.log"


DATA=/science/image/nlp-datasets/emanuele/data/conceptual_captions
#DATA=/home/mwp141/Working/volta/data/conceptual_captions
ANNOS=/home/mwp141/Prepared/iglue/volta/data/trans_cc/m2m-100-lg-seed-1337
FEATS=$DATA/resnet101_faster_rcnn_genome_imgfeats/volta
LANG_SAMPLING=/home/mwp141/Prepared/iglue/volta/data/trans_cc/p-lang-and-sent-alpha-0.3.npz
MODEL=ctrl_xuniter
MODEL_CONFIG=ctrl_xuniter_base
TETASK=Concap-pretrain-m2m-100-lg-sample
OUTPUT_DIR=checkpoints/conceptual_captions/pretrain/${MODEL}/${MODEL_CONFIG}/${TETASK}
LOGGING_DIR=logs/conceptual_captions/pretrain/${MODEL}/${MODEL_CONFIG}/${TETASK}


source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../volta
python train_concap_multilingual.py \
  --bert_model xlm-roberta-base \
  --from_pretrained xlm-roberta-base \
  --config_file config/${MODEL_CONFIG}.json \
  --train_batch_size 64 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 66 \
  --learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
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