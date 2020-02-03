"""Forward pass of the BNN """
import glob
import json
import os
import sys


# Necessary to run on CRC...
#os.chdir(os.environ['BASE_PATH'])
sys.path.append(os.environ['BASE_PATH'])

import numpy as np

from psych_metric.distrib.bnn.bnn_mcmc import BNNMCMC
from psych_metric.distrib.empirical_density import knn_density
from psych_metric.distrib.simplex import euclidean

from experiment import io
from experiment.research.bnn import proto_bnn_mcmc
from experiment.research.sjd import sjd_log_prob_exp


def load_sample_weights(weights_dir, filename='sampled_weights.json'):
    """Loads the sampled bnn weight sets from directory recursively into
    memory.
    """
    weights_sets = []

    # loop through json files, get weights, concatenate them
    # walk directory tree, grabbing all
    for filepath in glob.iglob(os.path.join(weights_dir, '*', filename)):
        with open(filepath, 'r') as f:
            weights = json.load(f)
            for vals in weights.values():
                if weights_sets:
                    for i, w in enumerate(vals['weights']):
                        weights_sets[i] = np.vstack((
                            weights_sets[i],
                            np.array(w)[vals['is_accepted']],
                        ))
                else:
                    weights_sets += [np.array(w)[vals['is_accepted']]
                        for w in vals['weights']]

    return weights_sets


def add_custom_args(parser):
    proto_bnn_mcmc.add_custom_args(parser)

    # add other args
    #parser.add_argument(
    #    '--bnn_weights_file',
    #    default=None,
    #    help='Path to the bnn weights file.',
    #)


if __name__ == '__main__':
    args = io.parse_args(
        ['sjd'],
        custom_args=add_custom_args,
        description='Runs KNNDE for euclidean BNN given the sampled weights.',
    )

    # Load sampled weights
    # combine sampled weights into a list

    # Load dataset's labels (given = target, conditionals = pred)
    if os.path.isfile(args.data.dataset_filepath):
        with open(args.data.dataset_filepath, 'r') as f:
            data = json.load(f)
            pred = np.array(data['output'], dtype=np.float32)
            givens = np.array(data['input'], dtype=np.float32)

        # load the euclidean simplex transform
        simplex_transform = euclidean.EuclideanSimplexTransform(pred.shape[1] + 1)
        simplex_transform.origin_adjust = np.array(data['origin_adjust'])
        simplex_transform.change_of_basis_matrix = np.array(
            data['change_of_basis'],
        )
        del data

        if os.path.isfile(args.bnn_weights_file):
            with open(args.bnn_weights_file, 'r') as f:
                weights_sets = [np.array(x, dtype=np.float32) for x in json.load(f)]
        elif os.path.isdir(args.bnn_weights_file):
            weights_sets = load_sample_weights(args.bnn_weights_file)
        else:
            raise ValueError('bnn weights file must be a file or dir.')
    else:
        raise ValueError('data dataset_filepath needs to be a file')

    # Create instance of BNNMCMC
    #bnn_mcmc = BNNMCMC(givens.shape[1], **vars(args.bnn))
    bnn_mcmc_args = vars(args.bnn)
    bnn_mcmc_args['dim'] = givens.shape[1]

    # Run KNNDE using BNNMCMC.predict(givens, weights)
    print('Perform KNNDE log prob on Train')
    log_probs = knn_density.euclid_bnn_knn_log_prob(
        givens,
        pred,
        simplex_transform,
        bnn_mcmc_args,
        weights_sets,
        args.sjd.knn_num_neighbors,
        args.sjd.n_jobs,
    )

    """
    print('Perform KNNDE log prob on Test')
    test_log_probs = knn_density.euclid_bnn_knn_log_prob(
        givens,
        pred,
        simplex_transform,
        bnn_mcmc,
        weights_sets,
        args.sjd.knn_num_neighbors,
        args.sjd.n_jobs,
    )

    test_log_probs = test_log_probs.sum()
    """

    io.save_json(
        args.output_dir,
        {
            'log_prob_sum': log_probs.sum(),
            'log_probs': log_probs
        },
    )
