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

import numpy as np

from experiment import io

# TODO load the BNN MCMC weights XOR load the BNN samples
if args.loaded_bnn_sampled_weights:
    # TODO fwd pass of BNN if loaded weights
else:
    #preds =

if args.target_is_predictor:
    # TODO load actual predictor's prediction

else:
    # TODO XOR load target label if Exp 2 where residuals is the measurement.


# TODO perform the measurement of euclidean distance on the BNN MCMC output to
# the actual prediction

# if np.linalg.norm can perform euclidean distance on [target, conditionals, classes] and output the euclidean distances of every [target, conditionals], then may just write that code here, instead of using the convenience psych_metric.metrics.measure().

differences = ta
for target_idx in range(len(targets)):


euclid_dists = np.linalg.norm(differences, axis=2)

if args.normalize:
    # Could calcuate the farthest euclidean distance of the probability
    # simplex (from one vertex to another) and then normalize the resulting
    # euclidean distances.
    vertex_1 = np.zeros(targets.shape[1])
    vertex_1[0] = 1

    vertex_2 = np.zeros(targets.shape[1])
    vertex_2[1] = 1

    prob_simplex_max_dist = np.linalg.norm(vertex_1 - vertex_2)

    euclid_dists /= prob_simplex_max_dist

# save Euclidean distances shape [target, conditionals]
np.savetxt(args.output_file, euclid_dists, delimiter=',')

# save summary of Euclidean distances:
# mean, min, max, median, 2 or 4 other quantiles.
euclid_dists_means = euclid_dists.mean(axis=1)

summary = {
    'summary_of_means': {
        'mean': euclid_dists_means.mean(),
        'max': euclid_dists_means.max(),
        'min': euclid_dists_means.min(),
        'median': np.median(euclid_dists_means),
    },
    'target_samples': {
        'mean': euclid_dists.mean(axis=1),
        'max': euclid_dists.max(axis=1),
        'min': euclid_dists.min(axis=1),
        'median': np.median(euclid_dists, axis=1),
    },
}

if args.quantiles_frac > 2:
    quantile_set = np.arange(1 + args.quantiles_frac) / args.quantiles_frac

    summary['summary_of_means']['quantiles'] = np.quantile(
        euclid_dists_means,
        quantile_set,
    )
    summary['target_samples']['quantiles'] = (
        euclid_dists_means,
        quantile_set,
    )

io.save_json(args.output_file.rpartion('.')[1] + '.json', summary)
