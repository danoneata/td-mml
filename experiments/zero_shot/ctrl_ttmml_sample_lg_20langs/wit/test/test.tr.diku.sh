#!/bin/bash
#SBATCH --job-name=tr-wit_lgsample
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --mem=45GB
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=6:00:00
#SBATCH --output="logs/wit-tr.sample_lg.step_100000.batch_size_256.langs_20.out"

LANG=tr
STRATEGY=sample_lg
STEP=100000
TRAIN_BATCH_SIZE=256
LANGS=20

echo "strategy:" $STRATEGY
echo "step:" $STEP
echo "train_batch_size (pretrain-cc)": $TRAIN_BATCH_SIZE
echo "lang":$LANG
echo "pretrain langs":$LANGS

TASK=20
MODEL=ctrl_xuniter
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=iglue_test_tasks_boxes36.diku
TRTASK=wit-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}
TETASK=RetrievalWIT${LANG}-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}
DIR=/science/image/nlp-datasets/tt-mml

PRETRAINED=$DIR/tt-mml-iglue/experiments/zero_shot/ctrl_ttmml_${STRATEGY}_${LANGS}langs/wit/train/${TRTASK}/RetrievalWIT_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=$DIR/tt-mml-iglue/experiments/zero_shot/ctrl_ttmml_${STRATEGY}_${LANGS}langs/wit/test/results/${MODEL_CONFIG}/${TETASK}

TEXT_PATH=$DIR/data/zero-shot/wit/annotations/test_${LANG}.jsonl
FEAT_PATH=$DIR/data/zero-shot/wit/features/wit_test_boxes36.lmdb

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../../volta

python eval_retrieval.py \
  --bert_model /science/image/nlp-datasets/tt-mml/huggingface/xlm-roberta-base \
  --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --caps_per_image 1 --split test_${LANG} --batch_size 1 \
  --output_dir ${OUTPUT_DIR} --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH}

deactivate
