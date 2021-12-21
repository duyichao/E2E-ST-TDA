#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: speech_to_text_tda_datatset.py
@time: 2021/5/17 7:58 afternoon
@desc: 
"""

import csv
import logging
import os.path as op
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig, get_features_or_waveform, _collate_frames

logger = logging.getLogger(__name__)


class SpeechToTextTdaDataset(FairseqDataset):
    def num_tokens_vec(self, indices):
        pass

    LANG_TAG_TEMPLATE = "<2{}>"

    def __init__(
            self,
            split: str,
            is_train_split: bool,
            data_cfg: S2TDataConfig,
            audio_paths: List[str],
            n_frames: List[int],
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            # src_tgt_texts: Optional[List[str]] = None,
            speakers: Optional[List[str]] = None,
            src_langs: Optional[List[str]] = None,
            tgt_langs: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            tgt_dict: Optional[Dictionary] = None,
            pre_tokenizer=None,
            bpe_tokenizer=None,
            use_consecutive_decoding=True,
            is_gen_split=False,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
                tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        # self.src_tgt_texts = src_tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.tgt_dict = tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer

        self.use_consecutive_decoding = use_consecutive_decoding
        self.is_gen_split = is_gen_split

        logger.info(self.__repr__())

    def __repr__(self):
        return (
                self.__class__.__name__
                + f'(split="{self.split}", n_samples={self.n_samples}, '
                  f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
                  f"shuffle={self.shuffle}, transforms={self.feature_transforms})"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    def tokenize_text(self, text: str):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        return text

    def __getitem__(
            self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        source = get_features_or_waveform(
            self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
        )
        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()

        target_mt, target_asr = None, None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target_mt = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target_mt = torch.cat((torch.LongTensor([lang_tag_idx]), target_mt), 0)
        if self.src_texts is not None:
            tokenized = self.tokenize_text(self.src_texts[index])
            target_asr = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.src_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target_asr = torch.cat((torch.LongTensor([lang_tag_idx]), target_asr), 0)

        return index, source, target_mt, target_asr,

    def __len__(self):
        return self.n_samples

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _ in samples], self.data_cfg.use_audio_input
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        target_asr_lengths, target_mt_lengths = None, None

        # if self.use_consecutive_decoding:
        if not self.is_gen_split:
            n_frames = torch.cat((n_frames, n_frames), 0)
            frames = torch.cat((frames, frames), 0)
            indices = torch.cat((indices, indices), 0)
            if self.tgt_texts is not None and self.src_texts is not None:
                target_asr_mt = fairseq_data_utils.collate_tokens(
                    [torch.cat((asr[:-1], mt), 0) for _, _, mt, asr in samples],  # asr[:-1], remove "eos" in transcript
                    self.tgt_dict.pad(),
                    self.tgt_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=False,
                ).index_select(0, order)
                target_mt_asr = fairseq_data_utils.collate_tokens(
                    [torch.cat((mt[:-1], asr), 0) for _, _, mt, asr in samples],
                    self.tgt_dict.pad(),
                    self.tgt_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=False,
                ).index_select(0, order)
                target = torch.cat((target_asr_mt, target_mt_asr), 0)  # TODO

                target_asr_lengths = torch.tensor(
                    [asr.size(0) for _, _, _, asr in samples], dtype=torch.long
                ).index_select(0, order)
                target_mt_lengths = torch.tensor(
                    [mt.size(0) for _, _, mt, _ in samples], dtype=torch.long
                ).index_select(0, order)
                target_lengths = target_mt_lengths + target_asr_lengths - 1
                target_lengths = torch.cat((target_lengths, target_lengths), 0)

                prev_output_asr_mt_tokens = fairseq_data_utils.collate_tokens(
                    [torch.cat((asr[:-1], mt), 0) for _, _, mt, asr in samples],
                    self.tgt_dict.pad(),
                    self.tgt_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=True,
                ).index_select(0, order)
                prev_output_mt_asr_tokens = fairseq_data_utils.collate_tokens(
                    [torch.cat((mt[:-1], asr), 0) for _, _, mt, asr in samples],
                    self.tgt_dict.pad(),
                    self.tgt_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=True,
                ).index_select(0, order)
                prev_output_tokens = torch.cat((prev_output_asr_mt_tokens, prev_output_mt_asr_tokens), 0)  # TODO
                ntokens = sum(mt.size(0) + asr.size(0) for _, _, mt, asr in samples)

                # target_asr_mt_token_index = torch.arange(max(target_lengths)).unsqueeze(0).repeat(target.size(0), 1)
        else:
            if self.tgt_texts is not None:
                target = fairseq_data_utils.collate_tokens(
                    [t for _, _, t, _ in samples],
                    self.tgt_dict.pad(),
                    self.tgt_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=False,
                )
                target = target.index_select(0, order)
                target_lengths = torch.tensor(
                    [t.size(0) for _, _, t, _ in samples], dtype=torch.long
                ).index_select(0, order)
                prev_output_tokens = fairseq_data_utils.collate_tokens(
                    [t for _, _, t, _ in samples],
                    self.tgt_dict.pad(),
                    self.tgt_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens = prev_output_tokens.index_select(0, order)
                ntokens = sum(t.size(0) for _, _, t, _ in samples)
                # print(target)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
            "target_asr_lengths": target_asr_lengths,
            "target_mt_lengths": target_mt_lengths,
        }
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(" "))
        return self.n_frames[index], t_len

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class SpeechToTextTdaDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

    @classmethod
    def _from_list(
            cls,
            split_name: str,
            is_train_split,
            samples: List[List[Dict]],
            data_cfg: S2TDataConfig,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_gen_split,
            # epoch,
    ) -> SpeechToTextTdaDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        src_tgt_texts = []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            # src_tgt_texts.extend(
            #     [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) + " " + ss[cls.KEY_TGT_TEXT] for ss in s]
            # )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
        return SpeechToTextTdaDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            # src_tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_gen_split=is_gen_split,
        )

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    @classmethod
    def from_tsv(
            cls,
            root: str,
            data_cfg: S2TDataConfig,
            splits: str,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split: bool,
            epoch: int,
            seed: int,
            is_gen_split: bool,
    ) -> SpeechToTextTdaDataset:
        # print(is_gen_split)
        samples = []
        _splits = splits.split(",")
        for split in _splits:
            tsv_path = op.join(root, f"{split}.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
                is_gen_split=is_gen_split,
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
