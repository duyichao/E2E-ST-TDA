Official implementation of AAAI'2022 paper "Regularizing End-to-End Speech Translation with Triangular Decomposition Agreement"

## Main Results

We evaluate the E2E-ST performance of our proposed approach (E2E-ST-TDA) on the MuST-C dataset with 8 languages, the results are as follows:

### ST Results
The case-sensitive BLEU scores on MuST-C tst-COMMON set. 

| Model          | Params. | Extra. | En-De | En-Fr | En-Ru | En-Es | En-It | En-Ro | En-Pt | En-Nl | Avg. |
| :------------- | :-----: | :----: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--: |
| E2E-ST-TDA$^s$ |   32M   |   ✗    | 24.3  | 34.6  | 15.9  | 28.3  | 24.2  | 23.4  | 30.3  | 28.7  | 26.2 |
| E2E-ST-TDA$^m$ |   76M   |   ✗    | 25.4  | 36.1  | 16.4  | 29.6  | 25.1  | 23.9  | 31.1  | 29.6  | 27.2 |
| E2E-ST-TDA$^m$ |   76M   |   ✔️    | 27.1  | 37.4  |   —   |   —   |   —   |   —   |   —   |   —   |  —   |

### ASR Results

The case-sensitive WER scores on MuST-C tst-COMMON set. 

| Model          | En-De | En-Fr | En-Ru | En-Es | En-It | En-Ro | En-Pt | En-Nl | Avg. |
| :------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--: |
| E2E-ST-TDA$^s$ | 16.4  | 15.6  | 16.6  | 16.4  | 16.2  | 16.6  | 16.9  | 16.2  | 16.4 |
| E2E-ST-TDA$^m$ | 14.9  | 14.1  | 15.7  | 14.4  | 15.2  | 15.4  | 16.5  | 14.9  | 15.1 |

## Requirements and Insallation

* python = 3.6
* pytorch = 1.8.1
* torchaudio = 0.8.1
* SoundFile = 0.10.3.post1
* numpy = 1.19.5
* omegaconf = 2.0.6
* PyYAML = 5.4.1
* sentencepiece = 0.1.96
* sacrebleu = 1.5.1

You can install this project by
```shell
cd E2E-ST-TDA
pip install --editable ./
```

## File structure

```text
fairseq
├── data
│    ├──/audio/
│        └── speech_to_text_tda_datasets.py
├── tasks
│    └── speech_to_text_tda.py
├── criterions
│    └── label_smoothed_cross_entropy_with_tda.py
├── dataclass
│    └── configs.py
├── /
fairseq_cli
├── generate_tda.py
├── /
myscripts
├── train_tda.sh
├── eval_tda.sh
```

## Instructions
### Preparations and Configurations
#### Pretrained ASR Model
The pre-trained ASR model can be found at [Fairseq S2T MuST-C Example](https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md).

#### Training Data

The TSV manifests we used are different from Fairseq S2T MuST-C Example, as follows:

``` tsv
id	audio	n_frames	speaker	src_lang	src_text	tgt_lang	tgt_text
ted_1_0	/data/share/ycdu/data/mustc/en-de/fbank80.zip:41:51328	160	spk.1	en	There was no motorcade back there.	de	Hinter mir war gar keine Autokolonne.
```

#### Config File

```yaml
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: /data/share/ycdu/data/mustc/en-de/kl_joint_data/spm_unigram10000_joint.model
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - utterance_cmvn
  _train:
  - utterance_cmvn
  - specaugment
vocab_filename: /data/share/ycdu/data/mustc/en-de/kl_joint_data/spm_unigram10000_joint.txt
prepend_tgt_lang_tag: True
```

## Training

```shell
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
  --tensorboard-logdir ${ST_SAVE_DIR}/tensorboard 
```

where `ST_SAVE_DIR`is the checkpoint root path. The ST encoder is pre-trained by ASR for faster training and better performance: `--load-pretrained-encoder-from <ASR_SAVE_DIR/ASR_CHECKPOINT_FILENAME>`. We set `--update-freq 4` to simulate 4 GPUs with 1 GPU.  We add the target language tag `<2de>/<2en>` as the target BOS to distinguish the `ST-BT` path and the `ASR-MT` path, specifically, we set `--ignore-prefix-size 1`.

## Inference

```shell
CHECKPOINT_FILENAME=tda_checkpoint_${MODEL_SIZE}.pt
python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

if [ ${GEN_TASK} == "st" ]; then
  if [ ${GEN_DIRECTION} = "asr_st" ]; then
    GEN_SUBSET=tst-COMMON_st1
  elif [ ${GEN_DIRECTION} = "st_asr" ]; then
    GEN_SUBSET=tst-COMMON_st
  fi
  CUDA_VISIBLE_DEVICES=${CUDA_IDS}, \
    python ${GEN_SCRIPTS}/generate_tda.py  ${MUSTC_ROOT}/en-${TGT_LANG}/kl_joint_data \
    --config-yaml config_joint.yaml --gen-subset ${GEN_SUBSET} --task speech_to_text_tda \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} --prefix-size 1 \
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
    python ${GEN_SCRIPTS}/generate_tda.py ${MUSTC_ROOT}/en-${TGT_LANG}/kl_joint_data \
    --config-yaml config_joint.yaml --gen-subset ${GEN_SUBSET} --task speech_to_text_tda \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --prefix-size 1 --max-tokens 50000 --scoring wer --wer-tokenizer 13a \
    --tda-task-type ${GEN_TASK} --tda-decoding-direction ${GEN_DIRECTION} \
    --quiet
fi
```

 For inference, we force decoding from the target language tag (as BOS) via `--prefix-size 1`. We also provide well-trained [models and vocabularies](https://drive.google.com/drive/folders/1WDgue_Bm1HxRmpKVf_mAmz0rbdUKQMox?usp=sharing) files for reproduction.