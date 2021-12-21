#!/usr/bin/env bash

CUDA_IDS=$1
TGT_LANG=$2
CKPT_DATE=$3
GEN_TASK=$4
GEN_DIRECTION=$5
MODEl_SIZE=$6

MUSTC_ROOT=/data/share/ycdu/data/mustc
ST_SAVE_DIR=/data/share/ycdu/st_log/${TGT_LANG}/st_tda_${MODEL_SIZE}_${CKPT_DATE}
CHECKPOINT_FILENAME=tda_checkpoint_${MODEL_SIZE}.pt

CLI_SCRIPTS=/data/ycdu/workspace/E2E-ST-TDA/fairseq_cli

python ../scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

if [ ${GEN_TASK} == "st" ]; then
  if [ ${GEN_DIRECTION} = "asr_st" ]; then
    GEN_SUBSET=tst-COMMON_st1
  elif [ ${GEN_DIRECTION} = "st_asr" ]; then
    GEN_SUBSET=tst-COMMON_st
  fi
  CUDA_VISIBLE_DEVICES=${CUDA_IDS}, \
    python ${CLI_SCRIPTS}/generate_tda.py ${MUSTC_ROOT}/en-${TGT_LANG}/kl_joint_data \
    --config-yaml config_joint.yaml --gen-subset ${GEN_SUBSET} --task speech_to_text_tda \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} --prefix-size 1 --speech-tgt-lang ${TGT_LANG} \
    --tda-task-type ${GEN_TASK} --tda-decoding-direction ${GEN_DIRECTION} \
    --max-tokens 50000 --beam 5 --scoring sacrebleu \
    --quiet
elif [ ${GEN_TASK} == "asr" ]; then
  if [ ${GEN_DIRECTION} = "asr_st" ]; then
    GEN_SUBSET=tst-COMMON_asr
  elif [ ${GEN_DIRECTION} = "st_asr" ]; then
    GEN_SUBSET=tst-COMMON_asr1
  fi
  CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
    python ${CLI_SCRIPTS}/generate_tda.py ${MUSTC_ROOT}/en-${TGT_LANG}/kl_joint_data \
    --config-yaml config_joint.yaml --gen-subset ${GEN_SUBSET} --task speech_to_text_tda \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} --speech-tgt-lang ${TGT_LANG} \
    --prefix-size 1 --max-tokens 50000 --scoring wer --wer-tokenizer 13a \
    --tda-task-type ${GEN_TASK} --tda-decoding-direction ${GEN_DIRECTION} \
    --quiet
fi
