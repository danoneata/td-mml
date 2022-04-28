#!/bin/bash
#SBATCH --job-name=de-xgqa_lgsample
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --mem=30GB
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=2:00:00
#SBATCH --output="logs/xgqa-de.sample_lg.step_100000.batch_size_256.langs_20.out"

LANG=de
STRATEGY=sample_lg
STEP=100000
LANGS=20
TRAIN_BATCH_SIZE=256
TASK_NAME=xgqa

echo "strategy:" $STRATEGY
echo "step:" $STEP
echo "train_batch_size (pretrain-cc)": $TRAIN_BATCH_SIZE
echo "lang":$LANG
echo "pretrain langs":$LANGS

TASK=15
MODEL=ctrl_xuniter
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=iglue_test_tasks_boxes36.diku
TRTASK=${TASK_NAME}-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}
TETASK=xGQA${LANG}-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}
DIR=/science/image/nlp-datasets/tt-mml

PRETRAINED=$DIR/tt-mml-iglue/experiments/zero_shot/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${TASK_NAME}/train/${TRTASK}/GQA_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=$DIR/tt-mml-iglue/experiments/zero_shot/ctrl_ttmml_${STRATEGY}_${LANGS}langs/${TASK_NAME}/test/results/${MODEL_CONFIG}/${TETASK}

TEXT_PATH=$DIR/data/zero-shot/xGQA/annotations/few_shot/${LANG}/test.json
FEAT_PATH=$DIR/data/zero-shot/gqa/features/vg-gqa_boxes36.lmdb

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../../volta

python eval_task.py \
  --bert_model /science/image/nlp-datasets/tt-mml/huggingface/xlm-roberta-base \
  --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --split test_${LANG} \
  --val_annotations_jsonpath ${TEXT_PATH} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
  --output_dir ${OUTPUT_DIR}

python scripts/GQA_score.py \
  --preds_file ${OUTPUT_DIR}/pytorch_model_best.bin-/test_${LANG}_result.json \
  --truth_file $TEXT_PATH
deactivate
