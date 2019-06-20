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


@register_criterion('lp_cross_entropy')
class LpCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--power', default=1., type=float, metavar='P',
        help='power')
        # fmt: on


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        logits = model(**sample['net_input'])[0]
        logits = logits.view(-1, logits.size(-1))
        Y = model.get_targets(sample, logits).view(-1)

        # for numerical stability
        logits_max = logits.max(1)[0].clone().detach()

        # exponentiate the logits
        logits_exp = torch.exp(logits - logits_max.view(Y.size(0), 1))
        # compute the normalization constant and detach it
        logits_Z = logits_exp.sum(1).clone().detach()
        # compute the actual probabilities
        pred = logits_exp / logits_Z.view(Y.size(0), 1)
        # prepare a detached softmax output
        pred_detach = pred.clone().detach()

        Y_onehot = F.one_hot(Y, num_classes=pred.size(1)).float()
        value = (torch.abs(Y_onehot - pred) ** (1+self.args.power))/(1+self.args.power)

        loss = (value / pred_detach).sum(1)
        
        if reduce:
            loss = loss.mean()

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

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
