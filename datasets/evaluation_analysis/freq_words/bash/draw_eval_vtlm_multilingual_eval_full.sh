#!/bin/bash
#SBATCH --job-name=vtlm_multilingual_eval-t_mar
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH -p gpu --gres=gpu
#SBATCH --time=2:00:00
#SBATCH --output="logs/draw.score_figs.multilingual_eval_full.vtlm.tt-mml"

LANGS=6
LANGS_SET=id-sw-ta-tr-en-zh
TASK=marvl
MODE=vtlm
EVAL_MODE=translate_train
TRANS_MODE=full
TRAIN_BATCH_SIZE=256
STEPS=100000
TASK_EVAL_TEST=MaRVL
PRETRAIN_MODEL=TTMML

DIR=/science/image/nlp-datasets/tt-mml
SCORE_PATH=$DIR/tt-mml-iglue/experiments/${EVAL_MODE}/ctrl_ttmml_${MODE}_20langs/${TASK}/test/analysis/frequence_analysis
FREQ_PATH=$DIR/tt-mml-iglue/datasets/evaluation_analyse/freq_words/data
FIGS_PATH=$DIR/tt-mml-iglue/datasets/evaluation_analyse/freq_words/figs

echo "strategy:" $MODE
echo "step:" $STEPS
echo "train_batch_size (pretrain-cc)": $TRAIN_BATCH_SIZE
echo "pretrain langs":$LANGS
cd ..

python draw.py \
  --langs $LANGS_SET \
  --task $TASK \
  --mode $MODE \
  --eval_mode $EVAL_MODE \
  --pretrain_model $PRETRAIN_MODEL \
  --task_eval_test $TASK_EVAL_TEST \
  --batch_size $TRAIN_BATCH_SIZE \
  --steps $STEPS \
  --pretrain_langs $LANGS \
  --freq_path $FREQ_PATH \
  --score_path $SCORE_PATH \
  --figs_path $FIGS_PATH \
  --translation_mode $TRANS_MODE

deactivate
