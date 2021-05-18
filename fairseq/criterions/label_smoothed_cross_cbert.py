# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedBERTCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

    autoregressive_mask: bool = field(
        default=True,
        metadata={"help": "allow autoregressive mask"},
    )
    lm_weight: float = field(
        default=0.5,
        metadata={"help": "epsilon for label smoothing, 0 means no bert loss"},
    )
    n_bert_update: int = field(
        default=1,
        metadata={"help": "number of bert updates"},
    )
    scheme: str = field(
        default='span',
        metadata={"help": "masking scheme"},
    )    


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
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
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_cbert", dataclass=LabelSmoothedBERTCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropycBERTCriterion(FairseqCriterion): ##
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        lm_weight=0.5,
        autoregressive_mask=False,
        n_bert_update=1,
        scheme=None,        
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.lm_weight = lm_weight
        self.autoregressive_mask = autoregressive_mask = True
        self.n_bert_update = n_bert_update
        self.iter = 0        

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss_tm, nll_loss_tm = self.compute_loss(model, net_output, sample, reduce=reduce)

        net_output_bert = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["span_input"])
        loss_lm, nll_loss_lm = self.compute_loss(model, net_output_bert, sample, reduce=reduce)
 
        loss = loss_tm + self.lm_weight * loss_lm 
        nll_loss = nll_loss_tm + self.lm_weight * nll_loss_lm      

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,

            'loss_tm': loss_tm.data,
            'nll_loss_tm': nll_loss_tm.data,

            'loss_lm': loss_lm.data,
            'nll_loss_lm': nll_loss_lm.data,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, isbert=False):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if isbert==True: # call sample["span_output"]
            target = model.get_span_targets(sample, net_output)
        else: # call sample["target"]
            target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, isbert=False):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, isbert=isbert)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None: ##
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))

        loss_sum_tm = sum(log.get("loss_tm", 0) for log in logging_outputs)
        nll_loss_sum_tm = sum(log.get("nll_loss_tm", 0) for log in logging_outputs)
        metrics.log_scalar("loss_tm", loss_sum_tm / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_tm", nll_loss_sum_tm / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived("ppl_tm", lambda meters: utils.get_perplexity(meters["nll_loss_tm"].avg))

        loss_sum_lm = sum(log.get("loss_lm", 0) for log in logging_outputs)
        nll_loss_sum_lm = sum(log.get("nll_loss_lm", 0) for log in logging_outputs)
        metrics.log_scalar("loss_lm", loss_sum_lm / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_lm", nll_loss_sum_lm / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived("ppl_lm", lambda meters: utils.get_perplexity(meters["nll_loss_lm"].avg))        

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


# import math
# from dataclasses import dataclass, field

# import torch
# from fairseq import metrics, utils
# from fairseq.criterions import FairseqCriterion, register_criterion
# from fairseq.dataclass import FairseqDataclass
# from omegaconf import II


# @dataclass
# class LabelSmoothedBERTCrossEntropyCriterionConfig(FairseqDataclass):
#     label_smoothing: float = field(
#         default=0.0,
#         metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
#     )
#     report_accuracy: bool = field(
#         default=False,
#         metadata={"help": "report accuracy metric"},
#     )
#     ignore_prefix_size: int = field(
#         default=0,
#         metadata={"help": "Ignore first N tokens"},
#     )
#     sentence_avg: bool = II("optimization.sentence_avg")

#     autoregressive_mask: bool = field(
#         default=True,
#         metadata={"help": "allow autoregressive mask"},
#     )
#     lm_weight: float = field(
#         default=0.5,
#         metadata={"help": "epsilon for label smoothing, 0 means no bert loss"},
#     )

# def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
#     if target.dim() == lprobs.dim() - 1:
#         target = target.unsqueeze(-1)
#     nll_loss = -lprobs.gather(dim=-1, index=target)
#     smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
#     if ignore_index is not None:
#         pad_mask = target.eq(ignore_index)
#         nll_loss.masked_fill_(pad_mask, 0.0)
#         smooth_loss.masked_fill_(pad_mask, 0.0)
#     else:
#         nll_loss = nll_loss.squeeze(-1)
#         smooth_loss = smooth_loss.squeeze(-1)
#     if reduce:
#         nll_loss = nll_loss.sum()
#         smooth_loss = smooth_loss.sum()
#     eps_i = epsilon / lprobs.size(-1)
#     loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
#     return loss, nll_loss


# @register_criterion(
#     "label_smoothed_cross_entropy_cbert", dataclass=LabelSmoothedBERTCrossEntropyCriterionConfig
# )
# class LabelSmoothedCrossEntropycBERTCriterion(FairseqCriterion): ##
#     def __init__(
#         self,
#         task,
#         sentence_avg,
#         label_smoothing,
#         ignore_prefix_size=0,
#         report_accuracy=False,
#         lm_weight=0.5,
#         autoregressive_mask=True,
#     ):
#         super().__init__(task)
#         self.sentence_avg = sentence_avg
#         self.eps = label_smoothing
#         self.ignore_prefix_size = ignore_prefix_size
#         self.report_accuracy = report_accuracy

#         self.lm_weight = lm_weight
#         self.autoregressive_mask = autoregressive_mask


#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.

#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """

#         net_output = model(**sample["net_input"])
#         loss_tm, nll_loss_tm = self.compute_loss(model, net_output, sample, reduce=reduce)

#         net_output_bert = model(src_tokens=sample["net_input"]["src_tokens"], src_lengths=sample["net_input"]["src_lengths"], prev_output_tokens=sample["span_input"])
#         loss_lm, nll_loss_lm = self.compute_loss(model, net_output_bert, sample, reduce=reduce, isbert=True) # False로 돌려짐..
 
#         loss = loss_tm + self.lm_weight * loss_lm 
#         nll_loss = nll_loss_tm + self.lm_weight * nll_loss_lm             

#         sample_size = (
#             sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
#         )
#         logging_output = {
#             "loss": loss.data,
#             "nll_loss": nll_loss.data,
#             "ntokens": sample["ntokens"],
#             "nsentences": sample["target"].size(0),
#             "sample_size": sample_size,

#             'loss_tm': loss_tm.data,
#             'nll_loss_tm': nll_loss_tm.data,

#             'loss_lm': loss_lm.data,
#             'nll_loss_lm': nll_loss_lm.data,
#         }
#         if self.report_accuracy:
#             n_correct, total = self.compute_accuracy(model, net_output, sample)
#             logging_output["n_correct"] = utils.item(n_correct.data)
#             logging_output["total"] = utils.item(total.data)
#         return loss, sample_size, logging_output

#     def get_lprobs_and_target(self, model, net_output, sample, isbert=False):
#         lprobs = model.get_normalized_probs(net_output, log_probs=True)
#         if isbert==True: # call sample["span_output"]
#             target = model.get_span_targets(sample, net_output)
#         else: # call sample["target"]
#             target = model.get_targets(sample, net_output)
#         if self.ignore_prefix_size > 0:
#             if getattr(lprobs, "batch_first", False):
#                 lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
#                 target = target[:, self.ignore_prefix_size :].contiguous()
#             else:
#                 lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
#                 target = target[self.ignore_prefix_size :, :].contiguous()
#         return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

#     def compute_loss(self, model, net_output, sample, reduce=True, isbert=False):
#         lprobs, target = self.get_lprobs_and_target(model, net_output, sample, isbert=isbert)
#         loss, nll_loss = label_smoothed_nll_loss(
#             lprobs,
#             target,
#             self.eps,
#             ignore_index=self.padding_idx,
#             reduce=reduce,
#         )
#         return loss, nll_loss

#     def compute_accuracy(self, model, net_output, sample):
#         lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
#         mask = target.ne(self.padding_idx)
#         n_correct = torch.sum(
#             lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
#         )
#         total = torch.sum(mask)
#         return n_correct, total

#     @classmethod
#     def reduce_metrics(cls, logging_outputs) -> None: ##
#         """Aggregate logging outputs from data parallel training."""
#         loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
#         nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
#         ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
#         sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
#         metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
#         metrics.log_scalar("nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
#         metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))

#         loss_sum_tm = sum(log.get("loss_tm", 0) for log in logging_outputs)
#         nll_loss_sum_tm = sum(log.get("nll_loss_tm", 0) for log in logging_outputs)
#         metrics.log_scalar("loss_tm", loss_sum_tm / sample_size / math.log(2), sample_size, round=3)
#         metrics.log_scalar("nll_loss_tm", nll_loss_sum_tm / ntokens / math.log(2), ntokens, round=3)
#         metrics.log_derived("ppl_tm", lambda meters: utils.get_perplexity(meters["nll_loss_tm"].avg))

#         loss_sum_lm = sum(log.get("loss_lm", 0) for log in logging_outputs)
#         nll_loss_sum_lm = sum(log.get("nll_loss_lm", 0) for log in logging_outputs)
#         metrics.log_scalar("loss_lm", loss_sum_lm / sample_size / math.log(2), sample_size, round=3)
#         metrics.log_scalar("nll_loss_lm", nll_loss_sum_lm / ntokens / math.log(2), ntokens, round=3)
#         metrics.log_derived("ppl_lm", lambda meters: utils.get_perplexity(meters["nll_loss_lm"].avg))        

#         total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
#         if total > 0:
#             metrics.log_scalar("total", total)
#             n_correct = utils.item(
#                 sum(log.get("n_correct", 0) for log in logging_outputs)
#             )
#             metrics.log_scalar("n_correct", n_correct)
#             metrics.log_derived(
#                 "accuracy",
#                 lambda meters: round(
#                     meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
#                 )
#                 if meters["total"].sum > 0
#                 else float("nan"),
#             )

#     @staticmethod
#     def logging_outputs_can_be_summed() -> bool:
#         """
#         Whether the logging outputs returned by `forward` can be summed
#         across workers prior to calling `reduce_metrics`. Setting this
#         to True will improves distributed training speed.
#         """
#         return True
