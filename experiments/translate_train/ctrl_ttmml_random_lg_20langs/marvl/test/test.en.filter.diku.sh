#!/bin/bash
#SBATCH --job-name=enfilter_mar_lgrandom
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=2:00:00
#SBATCH --output="logs/multilingual.marvl-en.random_lg.step_100000.batch_size_256.langs_20.eval_step2.filter.out"

LANG=en
STRATEGY=random_lg
STEP=100000
TRAIN_BATCH_SIZE=256
LANGS=20

EVAL_EPOCH=2
EVAL_STEP=24291

echo "strategy:" $STRATEGY
echo "pretrain step:" $STEP
echo "train_batch_size (pretrain-cc)": $TRAIN_BATCH_SIZE
echo "lang":$LANG
echo "pretrain langs":$LANGS
echo "iglue type: translation-based multilingual iglue"
echo "evaluation steps" $EVAL_STEP

TASK=12
MODEL=ctrl_xuniter
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=iglue_test_tasks_boxes36.diku
TRTASK=marvl-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}-filter
TETASK=MaRVL${LANG}-${STRATEGY}-batch_size_${TRAIN_BATCH_SIZE}-step_${STEP}-langs_${LANGS}-eval_step_${EVAL_STEP}-filter
DIR=/science/image/nlp-datasets/tt-mml

#PRETRAINED=$DIR/tt-mml-iglue/experiments/translate_train/ctrl_ttmml_${STRATEGY}_${LANGS}langs/marvl/train/${TRTASK}/NLVR2_${MODEL_CONFIG}/pytorch_model_best.bin
PRETRAINED=$DIR/tt-mml-iglue/experiments/translate_train/ctrl_ttmml_${STRATEGY}_${LANGS}langs/marvl/train/${TRTASK}/NLVR2_${MODEL_CONFIG}/pytorch_model_epoch_${EVAL_EPOCH}_step_${EVAL_STEP}.bin
OUTPUT_DIR=$DIR/tt-mml-iglue/experiments/translate_train/ctrl_ttmml_${STRATEGY}_${LANGS}langs/marvl/test/results/${MODEL_CONFIG}/${TETASK}

TEXT_PATH=$DIR/data/zero-shot/nlvr2/annotations/test.jsonl
FEAT_PATH=$DIR/data/zero-shot/nlvr2/features/nlvr2_boxes36.lmdb

source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../../volta

python eval_task.py \
  --bert_model /science/image/nlp-datasets/tt-mml/huggingface/xlm-roberta-base \
  --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
  --output_dir ${OUTPUT_DIR}

deactivate
