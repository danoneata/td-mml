#!/bin/bash
#SBATCH --job-name=en-t_mar_lgsample
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --mem=30GB
#SBATCH -p gpu --gres=gpu
#SBATCH --time=2:00:00
#SBATCH --output="log/marvl-en.sample_lg.step_100000.batch_size_256.langs_20.out"

LANG=en
STRATEGY=sample_lg
STEP=100000
TRAIN_BATCH_SIZE=256
LANGS=20

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



source /science/image/nlp-datasets/tt-mml/envs/tt-mml/bin/activate

cd ../../../../../volta

python eval_task.py \
  --bert_model /science/image/nlp-datasets/tt-mml/huggingface/xlm-roberta-base \
  --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
  --output_dir ${OUTPUT_DIR}

deactivate
