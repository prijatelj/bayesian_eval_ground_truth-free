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
from experiment.research.bnn.bnn_mcmc_fwd import load_sample_weights
from experiment.research.bnn import proto_bnn_mcmc

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
    parser.add_argument(
        '--loaded_bnn_outputs',
        default=None,
        action='store_true',
        help=' '.join([
            'Pass if the bnn_weights file is actually the bnn output, rather',
            'than the expected BNN weight sets. NOT IMPLEMNETED yet.',
        ])
    )

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


# TODO create argparser
args = io.parse_args(
    ['sjd'],
    custom_args=add_custom_args,
    description='Runs KNNDE for euclidean BNN given the sampled weights.',
)

output_dir = io.create_dirs(args.output_dir)

if os.path.isfile(args.data.dataset_filepath):
    with open(args.data.dataset_filepath, 'r') as f:
        data = json.load(f)
        pred = np.array(data['conditionals'], dtype=np.float32)
        givens = np.array(data['givens'], dtype=np.float32)

    # load the euclidean simplex transform
    simplex_transform = EuclideanSimplexTransform(pred.shape[1] + 1)
    simplex_transform.origin_adjust = np.array(data['origin_adjust'])
    # NOTE there is only a transpose for the older data.
    simplex_transform.change_of_basis_matrix = np.array(
        data['change_of_basis'],
    ).T
    del data

else:
    raise ValueError('data dataset_filepath needs to be a file')

if not args.target_is_task_target:
    # TODO Exp 1: load actual predictor's prediction
    targets = pred
    task_targets = givens
else:
    # TODO Exp 2: load target label if Exp 2 where measurement is residuals.
    targets = givens
    task_targets = targets
del pred
del givens

# TODO load the BNN MCMC weights XOR load the BNN samples
if not args.loaded_bnn_outputs:
    if task_targets is None:
        raise NotImplementedError()

    if os.path.isfile(args.bnn_weights_file):
        with open(args.bnn_weights_file, 'r') as f:
            weights_sets = [
                np.array(x, dtype=np.float32) for x in json.load(f)
            ]
    elif os.path.isdir(args.bnn_weights_file):
        weights_sets = load_sample_weights(args.bnn_weights_file)
    else:
        raise ValueError('bnn weights file must be a file or dir.')

    # Create instance of BNNMCMC
    bnn_mcmc_args = vars(args.bnn)
    bnn_mcmc_args['dim'] = task_targets.shape[1]
    bnn_mcmc_args['sess_config'] = io.get_tf_config(
        args.cpu_cores,
        args.cpu,
        args.gpu,
    )

    # TODO fwd pass of BNN if loaded weights
    bnn = BNNMCMC(**bnn_mcmc_args)
    preds = bnn.predict(task_targets, weights_sets)
else:
    # if not loading the BNN sampled weights sets, then loading the BNN output
    # which if saved would be hdf5
    #preds =
    raise NotImplementedError('ATM, does not expect BNN output do be given')


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
# preds is no longer used from this point on, and it has been modified
del preds

euclid_dists = np.linalg.norm(differences, axis=2)

if args.normalize:
    # Normalizes by the largest possible distance within the probability
    # simplex, which is the distance form one vertex to any other vertex
    # because the probability simplex is regular (ie. 2-simplex is a
    # equilateral triangle).
    euclid_dists /= np.sqrt(2)

# Save Euclidean distances shape [target, conditionals]
np.savetxt(
    os.path.join(output_dir, 'euclid_dists.csv'),
    euclid_dists,
    delimiter=',',
)

# save summary of Euclidean distances:
if args.quantiles_frac > 2:
    quantile_set = np.arange(1 + args.quantiles_frac) / args.quantiles_frac
else:
    quantile_set = None

def summary_arr(arr, quantile_set=None, axis=None):
    summary = {
        'mean': np.mean(arr, axis=axis),
        'max': np.max(arr, axis=axis),
        'min': np.min(arr, axis=axis),
        'median': np.median(arr, axis=axis),
    }

    if quantile_set is not None:
        summary['quantile'] = np.quantile(arr, quantile_set, axis=axis)

    return summary

summary = {
    'overview': summary_arr(euclid_dists, quantile_set=quantile_set),
    'summary_of_means': summary_arr(euclid_dists.mean(axis=1), quantile_set),
    'summary_of_maxs': summary_arr(euclid_dists.max(axis=1), quantile_set),
    'summary_of_mins': summary_arr(euclid_dists.min(axis=1), quantile_set),
    'summary_of_medians': summary_arr(
        np.median(euclid_dists, axis=1),
        quantile_set,
    ),
    'target_samples': summary_arr(euclid_dists, quantile_set, axis=1),
}

io.save_json(os.path.join(output_dir, 'euclid_dists_summary.json'), summary)
