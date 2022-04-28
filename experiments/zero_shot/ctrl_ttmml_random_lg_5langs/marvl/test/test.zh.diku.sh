#!/bin/bash
#SBATCH --job-name=zh-t_mar_random_lg
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --mem=30GB
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=2:00:00
#SBATCH --output="logs/marvl-zh.random_lg.step_100000.batch_size_256.langs_5.out"

LANG=zh
STRATEGY=random_lg
STEP=100000
TRAIN_BATCH_SIZE=256
LANGS=5

echo "strategy:" $STRATEGY
echo "step:" $STEP
echo "train_batch_size (pretrain-cc)": $TRAIN_BATCH_SIZE
echo "lang":$LANG
echo "pretrain langs":$LANGS

TASK=12
MODEL=ctrl_xuniter
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=iglue_test_tasks_boxes36.diku
TRTASK=marvl-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}
TETASK=MaRVL${LANG}-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}
DIR=/science/image/nlp-datasets/tt-mml

PRETRAINED=$DIR/tt-mml-iglue/experiments/zero_shot/ctrl_ttmml_${STRATEGY}_${LANGS}langs/marvl/train/${TRTASK}/NLVR2_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=$DIR/tt-mml-iglue/experiments/zero_shot/ctrl_ttmml_${STRATEGY}_${LANGS}langs/marvl/test/results/${MODEL_CONFIG}/${TETASK}

TEXT_PATH=$DIR/data/zero-shot/marvl/annotations/marvl-${LANG}.jsonl
FEAT_PATH=$DIR/data/zero-shot/marvl/features/marvl-${LANG}_boxes36.lmdb

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../../volta

python eval_task.py \
  --bert_model /science/image/nlp-datasets/tt-mml/huggingface/xlm-roberta-base \
  --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test \
  --output_dir ${OUTPUT_DIR}

deactivate
