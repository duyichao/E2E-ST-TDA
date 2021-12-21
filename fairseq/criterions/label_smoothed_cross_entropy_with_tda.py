#!/usr/bin/env/python
"""
@author: ycdu
@mail: ycdu666@gmail.com
@IDE: PyCharm
@file: label_smoothed_cross_entropy_with_tda.py
@time: 2021/5/18 9:05 afternoon
@desc: 
"""

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, \
    LabelSmoothedCrossEntropyCriterionConfig


def label_smoothed_nll_loss_(
        lprobs,
        target,
        epsilon,
        ignore_index=None,
        reduce=True
):
    lprobs, target = lprobs.view(-1, lprobs.size(-1)), target.view(-1)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy_with_tda", dataclass=LabelSmoothedCrossEntropyCriterionConfig)
class LabelSmoothedCrossEntropyCriterionWithTda(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.sentence_kl_flag = getattr(task.args, "sentence_level_kl_loss", False)
        self.word_kl_flag = getattr(task.args, "word_level_kl_loss", False)
        self.tgt_lang = getattr(task.args, "speech_tgt_lang", "tgt")
        self.sentence_kl_lambda = getattr(task.args, "sentence_kl_lambda", 1.0)
        self.word_kl_lambda = getattr(task.args, "word_kl_lambda", 1.0)
        # self.kl_loss_func = torch.nn.KLDivLoss(reduction='sum', log_target=True)
        self.kl_loss_func = torch.nn.KLDivLoss(reduction='none', log_target=True)
        self.mse_loss_func = torch.nn.MSELoss(reduction="sum")
        self.special_tokens = [
            self.task.tgt_dict.pad(),
            self.task.tgt_dict.eos(),
            self.task.tgt_dict.index("<2{}>".format(self.tgt_lang)),  # TODO
            self.task.tgt_dict.index("<2en>"),
        ]

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, smoothed_loss, nll_loss, word_kl_loss, sentence_kl_loss = \
            self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "smoothed_loss": smoothed_loss.data,
            "word_kl_loss": word_kl_loss.data,
            "sentence_kl_loss": sentence_kl_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target_(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        lprobs_asr_mt, lprobs_mt_asr = None, None
        if self.word_kl_flag:
            target_asr_lengths, target_mt_lengths = sample['target_asr_lengths'], sample['target_mt_lengths']
            lprobs_asr_mt = lprobs[:int((lprobs.size(0) / 2)), :, :]  # asr_mt
            _lprobs = lprobs[int(lprobs.size(0) / 2):, :, :]  # mt_asr
            lprobs_mt_asr = _lprobs.clone()
            for idx, asr_len in enumerate(target_asr_lengths):
                mt_len = target_mt_lengths[idx]
                # -1因为len包含了eos
                lprobs_mt_asr[idx, :asr_len - 1, :] = _lprobs[idx, mt_len - 1:mt_len - 1 + asr_len - 1, :]
                lprobs_mt_asr[idx, asr_len - 1:asr_len - 1 + mt_len - 1, :] = _lprobs[idx, :mt_len - 1, :]
            if self.ignore_prefix_size > 0:
                if getattr(lprobs, "batch_first", False):
                    lprobs_asr_mt = lprobs_asr_mt[:, self.ignore_prefix_size:, :].contiguous()
                    lprobs_mt_asr = lprobs_mt_asr[:, self.ignore_prefix_size:, :].contiguous()
                else:
                    lprobs_asr_mt = lprobs_asr_mt[self.ignore_prefix_size:, :, :].contiguous()
                    lprobs_mt_asr = lprobs_mt_asr[self.ignore_prefix_size:, :, :].contiguous()
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs, target, lprobs_asr_mt, lprobs_mt_asr

    def compute_loss(self, model, net_output, sample, reduce=True):
        sentence_kl_loss, word_kl_loss = torch.tensor(0.), torch.tensor(0.)
        lprobs, target, lprobs_asr_mt, lprobs_mt_asr = self.get_lprobs_and_target_(model, net_output, sample)
        smoothed_loss, nll_loss = label_smoothed_nll_loss_(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        loss = smoothed_loss
        if self.word_kl_flag:
            word_kl_loss = self.compute_word_kl_loss(
                target[:int(target.size(0) / 2), :],
                lprobs_asr_mt, lprobs_mt_asr,
                ignore_index=self.special_tokens
            )
            loss = loss + self.word_kl_lambda * word_kl_loss
        if self.sentence_kl_flag:
            sentence_kl_loss = self.compute_sentence_kl_loss(
                target,
                sample["target_lengths"],
                lprobs,
                self.padding_idx,
            )
            loss = loss + self.sentence_kl_lambda * sentence_kl_loss
        return loss, smoothed_loss, nll_loss, word_kl_loss, sentence_kl_loss

    def compute_sentence_kl_loss(self, target, target_lengths, lprobs, ignore_index):
        bsz = lprobs.size(0)
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        lprobs = lprobs.gather(dim=-1, index=target)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            lprobs.masked_fill_(pad_mask, 0.0)
        # sen_log_probs = torch.exp(lprobs.squeeze(-1).sum(dim=-1)/target_lengths)
        sen_log_probs = lprobs.squeeze(-1).sum(dim=-1)
        sentence_kl_loss = \
            self.mse_loss_func(sen_log_probs[:int(bsz / 2)], sen_log_probs[int(bsz / 2):])
        return sentence_kl_loss

    def compute_word_kl_loss(self, target, lprobs_asr_mt, lprobs_mt_asr, ignore_index):
        # if ignore_index is not None:
        #     pad_mask = lprobs_asr_mt.eq(ignore_index)
        #     lprobs_asr_mt.masked_fill_(pad_mask, 0.0)
        #     pad_mask = lprobs_mt_asr.eq(ignore_index)
        #     lprobs_mt_asr.masked_fill_(pad_mask, 0.0)
        # kl_loss = self.kl_loss_func(lprobs_asr_mt, torch.exp(lprobs_mt_asr)) + \
        #           self.kl_loss_func(lprobs_mt_asr, torch.exp(lprobs_asr_mt))
        word_kl_loss = self.kl_loss_func(lprobs_asr_mt, lprobs_mt_asr) + \
                       self.kl_loss_func(lprobs_mt_asr, lprobs_asr_mt)
        if ignore_index is not None:
            for ig_index in ignore_index:
                pad_mask = target.eq(ig_index).unsqueeze(2)
                word_kl_loss = word_kl_loss.masked_fill(pad_mask, 0.0)

        word_kl_loss = torch.sum(word_kl_loss)
        return word_kl_loss

    def compute_accuracy(self, model, net_output, sample):
        # lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # mask = target.ne(self.padding_idx)
        # n_correct = torch.sum(
        #     lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        # )
        # total = torch.sum(mask)
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # lprobs, target = lprobs.view(-1, lprobs.size(-1)), target.view(-1)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # loss, smoothed_loss, nll_loss, word_kl_loss, sentence_kl_loss
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        smoothed_loss_sum = sum(log.get("smoothed_loss", 0) for log in logging_outputs)
        word_kl_loss_sum = sum(log.get("word_kl_loss", 0) for log in logging_outputs)
        sentence_kl_loss_sum = sum(log.get("sentence_kl_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "smoothed_loss", smoothed_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "word_kl_loss", word_kl_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "sentence_kl_loss", sentence_kl_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
