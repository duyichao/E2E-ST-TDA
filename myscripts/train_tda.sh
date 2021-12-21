#!/usr/bin/env bash

CUDA_IDS=$1
TGT_LANG=$2
CKPT_DATE=$3
MODEL_SIZE=$4 # s:256, m:512, l:768, xl:1024

MUSTC_ROOT=/data/share/ycdu/data/mustc
ASR_SAVE_DIR=/data/share/ycdu/st_log/${TGT_LANG}/asr_path
ASR_CHECKPOINT_FILENAME=asr_checkpoint_${MODEL_SIZE}.pt
ST_SAVE_DIR=/data/share/ycdu/st_log/${TGT_LANG}/st_tda_${MODEL_SIZE}_${CKPT_DATE}
if [ ! -d $ST_SAVE_DIR ]; then
  mkdir $ST_SAVE_DIR -p
fi

CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
  fairseq-train ${MUSTC_ROOT}/en-${TGT_LANG}/kl_joint_data \
  --config-yaml config_joint.yaml --train-subset train_joint --valid-subset dev_joint \
  --save-dir ${ST_SAVE_DIR} --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${ASR_CHECKPOINT_FILENAME} \
  --criterion label_smoothed_cross_entropy_with_tda --report-accuracy --label-smoothing 0.1 --ignore-prefix-size 1 \
  --task speech_to_text_tda --arch s2t_transformer_tda_${MODEL_SIZE} --encoder-freezing-updates 0 \
  --update-freq 4 --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --num-workers 0 --max-tokens 10000 --max-epoch 150 \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 \
  --word-level-kl-loss --word-kl-lambda 1.0 --speech-tgt-lang ${TGT_LANG} \
  --tensorboard-logdir ${ST_SAVE_DIR}/tensorboard \
  >${ST_SAVE_DIR}/${TGT_LANG}-tda-${MODEL_SIZE}.log 2>&1
# --sentence-level-kl-loss --sentence-kl-lambda 1.0 \
# --kl-schedule-type cyclical --kl-cyclical-max-epoch 120 \
