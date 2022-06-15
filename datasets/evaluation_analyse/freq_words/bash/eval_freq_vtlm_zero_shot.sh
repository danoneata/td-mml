#!/bin/bash
#SBATCH --job-name=ta-t_mar_vtlm
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH -p gpu --gres=gpu
#SBATCH --time=2:00:00
#SBATCH --output="logs/zero_shot.vtlm.tt-mml.freq_analysis"

LANGS=6
LANGS_SET=en-id-sw-ta-tr-zh
TASK=marvl
MODE=vtlm
EVAL_MODE=zero_shot
TRAIN_BATCH_SIZE=256
STEPS=100000
TASK_EVAL_TEST=MaRVL
PRETRAIN_MODEL=TTMML

DIR=/science/image/nlp-datasets/tt-mml
ANNOTATOIN_PATH=$DIR/data/conceptual_captions/annotations/langs_20/m2m-100-lg-filtered
EVAL_ANNOTATION_PTAH=$DIR/data/zero-shot/marvl/annotations
EVAL_TEST_PATH=$DIR/tt-mml-iglue/experiments/${EVAL_MODE}/ctrl_ttmml_${MODE}_20langs/${TASK}/test/results/ctrl_xuniter_base
SCORE_OUTPUT_PATH=$DIR/tt-mml-iglue/experiments/${EVAL_MODE}/ctrl_ttmml_${MODE}_20langs/${TASK}/test/analysis/frequence_analysis

FREQ_OUTPUT_PATH=$DIR/tt-mml-iglue/datasets/evaluation_analyse/freq_words/data
EVAL_EN_PATH=$DIR/data/zero-shot/nlvr2/annotations

echo "strategy:" $MODE
echo "step:" $STEPS
echo "train_batch_size (pretrain-cc)": $TRAIN_BATCH_SIZE
echo "pretrain langs":$LANGS

cd ..

python freq_preprocess.py \
  --langs $LANGS_SET \
  --task $TASK \
  --mode $MODE \
  --eval_mode $EVAL_MODE \
  --pretrain_model $PRETRAIN_MODEL \
  --task_eval_test $TASK_EVAL_TEST \
  --batch_size $TRAIN_BATCH_SIZE \
  --steps $STEPS \
  --pretrain_langs $LANGS \
  --annotation_path $ANNOTATOIN_PATH \
  --eval_annotation_path $EVAL_ANNOTATION_PTAH \
  --eval_en_path $EVAL_EN_PATH \
  --eval_test_path $EVAL_TEST_PATH \
  --freq_output_path $FREQ_OUTPUT_PATH \
  --score_output_path $SCORE_OUTPUT_PATH

deactivate
