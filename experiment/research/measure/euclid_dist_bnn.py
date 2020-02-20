"""Performs the Experiment 1 of paper that confirms that the BNN MCMC captures
the conditional property of the predictor's predictions given the target label.

Notes
-----
Experiment 1 when target is actual predictor's prediction:
    Performing this where the target is the predictor's actual prediciton and
    the preds is the BNN MCMC outputs is performing experiment 1, for
    comparison to iid distributions to show our method captures the conditional
    relationship between (implictly showing this, explicitly showing out method
    closer matches the actual predictor's predictions than the iid methods.

Experiement 2 for residuals when target is given target label to predictor:
    Performing this where the target is the actual target label of the task and
    pred is the estimated predictions of the predictor via the BNN MCMC
    generates the distribution of residuals, which is a distribution of a
    measure and part of experiment 2.
"""
import csv
import json
import os

import numpy as np

from experiment import io
from experiment.research.bnn.bnn_mcmc_fwd import load_bnn_fwd
from experiment.research.bnn import proto_bnn_mcmc
from experiment.research.measure.measure import save_measures

from psych_metric.distrib.bnn.bnn_mcmc import BNNMCMC
from psych_metric.distrib.simplex.euclidean import EuclideanSimplexTransform

def add_custom_args(parser):
    proto_bnn_mcmc.add_custom_args(parser)

    # add other args
    parser.add_argument(
        '--target_is_task_target',
        default=None,
        action='store_true',
        help=' '.join([
            'Pass if target is task\'s target labels (Exp 2 where measure is',
            'residuals), rather than where the predictor\'s predictions are',
            'the target (Exp 1).',
        ])
    )

    """
    parser.add_argument(
        '--loaded_bnn_outputs',
        default=None,
        action='store_true',
        help=' '.join([
            'Pass if the bnn_weights file is actually the bnn output, rather',
            'than the expected BNN weight sets. NOT IMPLEMNETED yet.',
        ])
    )
    """

    parser.add_argument(
        '--normalize',
        default=None,
        action='store_true',
        help=' '.join([
            'Pass if the Euclidean distances are to be normalized by the',
            'largest distance between two points possible within the',
            'probability simplex (aka distance between two vertices of the',
            'probability simplex).',
        ])
    )

    parser.add_argument(
        '--quantiles_frac',
        default=5,
        type=int,
        help=' '.join([
            'The number of fractions 1 to serve as quantiles to be used. ie.',
            'quantile of 4 results in 0 (min), 0.25, 0.5 (median), 0.75  and',
            '1 (max). The min, max, and median are always calculated.',
            'Quantiles_frac less than or equal to 2 results in no other',
            'quantiles.',
        ])
    )


# Create argparser
args = io.parse_args(
    ['sjd'],
    custom_args=add_custom_args,
    description='Runs KNNDE for euclidean BNN given the sampled weights.',
)

output_dir = io.create_dirs(args.output_dir)

# Manage bnn mcmc args from argparse
bnn_mcmc_args = vars(args.bnn)
bnn_mcmc_args['sess_config'] = io.get_tf_config(
    args.cpu_cores,
    args.cpu,
    args.gpu,
)

givens, pred, weights_sets, bnn = load_bnn_fwd(
    args.data.dataset_filepath,
    args.bnn_weights_file,
    bnn_mcmc_args,
)

if not args.target_is_task_target:
    # Exp 1: predictor's prediction is the target: t(y) = y_hat
    targets = pred
else:
    # Exp 2: Original target label is target where measurement is residuals.
    targets = givens
del pred

# fwd pass of BNN if loaded weights
preds = bnn.predict(givens, weights_sets)

# Perform the measurement of euclidean distance on the BNN MCMC output to the
# actual prediction

# if np.linalg.norm can perform euclidean distance on [target, conditionals,
# classes] and output the euclidean distances of every [target, conditionals],
# then may just write that code here, instead of using the convenience
# psych_metric.metrics.measure().

for target_idx in range(len(targets)):
    # NOTE assumes shape of [targets, conditionals, classes]
    preds[target_idx] = targets[target_idx] - preds[target_idx]
differences = preds
# preds is no longer used from this point on, as it has been modified
del preds

euclid_dists = np.linalg.norm(differences, axis=2)

if args.normalize:
    # Normalizes by the largest possible distance within the probability
    # simplex, which is the distance form one vertex to any other vertex
    # because the probability simplex is regular (ie. 2-simplex is a
    # equilateral triangle).
    euclid_dists /= np.sqrt(2)

save_measures(output_dir, 'euclid_dists', euclid_dists, args.quantiles_frac)
