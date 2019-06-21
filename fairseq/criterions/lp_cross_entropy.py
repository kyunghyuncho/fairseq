# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion


@register_criterion('lp_cross_entropy')
class LpCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        self.ce = CrossEntropyCriterion(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--power', default=1., type=float, metavar='P', help='power')
        # fmt: on


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx).squeeze()

        logits = net_output[0]

        if logits.requires_grad:
            def logit_grad_hook(grad):
                osize = grad.size()
                grad_ = grad.view(-1, osize[-1])

                grad_pos = grad_.gt(0.).float()
                grad_neg = 1.-grad_pos
                prob_ = grad_pos * (1.-grad_) + grad_neg * (-grad_)

                #grad_vol = grad_.abs().sum(1, keepdim=True)
                grad_sign = grad_.sign()
                grad_mag = grad_.abs()

                #logits = grad_mag
                logits = prob_
                logits_max = logits.max(1, keepdim=True)[0]
                logits_exp = logits - logits_max
                logits_exp = torch.exp((self.args.power-1.) * logits_exp)
                logits_Z = logits_exp.sum(1, keepdim=True)
                grad_mag_weight = logits_exp / logits_Z

                grad_pow = grad_mag * grad_mag_weight 
                grad_pow = grad_pow / (1e-4 + torch.sqrt((grad_pow ** 2).sum(1, keepdim=True))) * torch.sqrt((grad_mag ** 2).sum(1, keepdim=True))

                grad_ = grad_pow * grad_sign
                return grad_.view(osize)

            logits.register_hook(logit_grad_hook)

        nll_loss, _ = self.ce.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        return nll_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
