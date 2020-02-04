"""Forward pass of the BNN """
import json
import logging
import os
from pathlib import Path
import sys


# Necessary to run on CRC...
#os.chdir(os.environ['BASE_PATH'])
try:
    sys.path.append(os.environ['BASE_PATH'])
except:
    logging.warning('environment variable `BASE_PATH` is not available; not appending anything to the system path.')


import numpy as np

from psych_metric.distrib.bnn.bnn_mcmc import BNNMCMC
from psych_metric.distrib.empirical_density import knn_density
from psych_metric.distrib.simplex import euclidean

from experiment import io
from experiment.research.bnn import proto_bnn_mcmc
from experiment.research.sjd import sjd_log_prob_exp


def load_sample_weights(weights_dir, filename='.json'):
    """Loads the sampled bnn weight sets from directory recursively into
    memory.
    """
    weights_sets = []

    # loop through json files, get weights, concatenate them
    # walk directory tree, grabbing all
    #for filepath in glob.iglob(os.path.join(weights_dir, '*', filename)):
    for filepath in Path(weights_dir).rglob('*' + filename):
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
        # NOTE there is only a transpose for the older data.
        simplex_transform.change_of_basis_matrix = np.array(
            data['change_of_basis'],
        ).T
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
    bnn_mcmc_args['sess_config'] = io.get_tf_config(
        #args.cpu_cores,
        1,
        args.cpu,
        args.gpu,
    )

    # Run KNNDE using BNNMCMC.predict(givens, weights)
    logging.info('Perform KNNDE log prob on Train')
    #log_probs = knn_density.euclid_bnn_knn_log_prob(
    log_probs = knn_density.euclid_bnn_knn_log_prob_sequence(
        givens,
        pred,
        simplex_transform,
        bnn_mcmc_args,
        weights_sets,
        args.sjd.knn_num_neighbors,
        False, # needs_transformed: No for prototyping
        args.sjd.n_jobs,
    )

    io.save_json(
        args.output_dir,
        {
            'log_prob_sum': log_probs.sum(),
            'log_probs': log_probs
        },
    )
