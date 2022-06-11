#!/bin/bash
#SBATCH --job-name=mar_lgrandom
#SBATCH --ntasks=1 --cpus-per-task=40
#SBATCH -p gpu --gres=gpu:titanrtx:1 --mem=5GB
#SBATCH --time=3-00:00:00
#SBATCH --output="logs/train.multilingual.marvl.random_lg.step_100000.batch_size_256.langs_20.full.log"

TASK=12
TASK_NAME=marvl
LANGS=6
LANGS_SET=id-sw-ta-tr-zh-en
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=iglue_trainval_tasks_boxes36.dtu
DIR=/science/image/nlp-datasets/tt-mml

STRATEGY=random_lg
STEP=100000
LANGS=20
TRAIN_BATCH_SIZE=256
PRETRAIN_FILE=pytorch_model_epoch_9_step_100000.bin
TETASK=${TASK_NAME}-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}-full

PRETRAINED=$DIR/checkpoints/iglue/pretrain/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${MODEL_CONFIG}/train_batch_size_${TRAIN_BATCH_SIZE}/conceptual_captions-${STRATEGY}/${MODEL_CONFIG}/${PRETRAIN_FILE}
OUTPUT_DIR=$DIR/tt-mml-iglue/experiments/translate_train/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${TASK_NAME}/train/${TETASK}
LOGGING_DIR=$DIR/logs/multilingual_marvl/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${MODEL_CONFIG}/train_batch_size_${TRAIN_BATCH_SIZE}/${TETASK}

TRANS_PATH=$DIR/data/translate_train/nlvr2/full

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../../volta
python train_task_multilingual.py \
    --bert_model /science/image/nlp-datasets/tt-mml/huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
    --from_pretrained ${PRETRAINED} --cache 500 \
    --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --gradient_accumulation_steps 4 --num_workers 10 --num_val_workers 10 \
    --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --logdir ${LOGGING_DIR} \
    --langs $LANGS_SET \
    --translation_path $TRANS_PATH \
    --num_workers 5

deactivate
