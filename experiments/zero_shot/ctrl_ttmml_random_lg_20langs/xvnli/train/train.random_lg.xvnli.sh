#!/bin/bash
#SBATCH --job-name=xvnli_random_lg
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --mem=30GB
#SBATCH -p gpu --gres=gpu
#SBATCH --time=1-12:00:00
#SBATCH --output="logs/train.xvnli.random_lg.step_100000.batch_size_256.langs_20.log"

TASK=19
TASK_NAME=xvnli
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=iglue_trainval_tasks_boxes36.dtu
DIR=/science/image/nlp-datasets/tt-mml

STRATEGY=random_lg
STEP=100000
LANGS=20
TRAIN_BATCH_SIZE=256
PRETRAIN_FILE=pytorch_model_epoch_9_step_100000.bin
TETASK=${TASK_NAME}-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}

PRETRAINED=$DIR/checkpoints/iglue/pretrain/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${MODEL_CONFIG}/train_batch_size_${TRAIN_BATCH_SIZE}/conceptual_captions-${STRATEGY}/${MODEL_CONFIG}/${PRETRAIN_FILE}
OUTPUT_DIR=$DIR/tt-mml-iglue/experiments/zero_shot/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${TASK_NAME}/train/${TETASK}
LOGGING_DIR=$DIR/logs/marvl/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${MODEL_CONFIG}/train_batch_size_${TRAIN_BATCH_SIZE}/${TETASK}

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../../volta
python train_task.py \
    --bert_model /science/image/nlp-datasets/tt-mml/huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
    --from_pretrained ${PRETRAINED} \
    --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --gradient_accumulation_steps 4 \
    --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --logdir ${LOGGING_DIR} \
#    --resume_file ${OUTPUT_DIR}/RetrievalFlickr30k_${MODEL_CONFIG}/pytorch_ckpt_latest.tar

deactivate
