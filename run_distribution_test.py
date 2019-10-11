"""Run the distribution tests."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import json
import logging
import os

import numpy as np
import tensorflow as tf

import experiment.io
from psych_metric import distribution_tests
from predictors import load_prep_data


def test_human_distrib(
    dataset_id,
    data_args,
    distrib_args,
    label_src,
    parts,
    mle_args=None,
    info_criterions=['bic'],
    random_seed=None,
):
    """Hypothesis testing of distributions of the human annotations.

    Parameters
    ----------
    dataset_id : str
        Identifier of dataset.
    data_args : dict
        Dict containing information pertaining to the data.
    distrib_args : dict
        Dict of str distrib ids to their initial parameters
    random_seed : int, optional
        Integer to use as the random seeds for MLE.
    """

    # TODO First, handle distrib test of src annotations
    # Load the src data, reformat as was done in training in `predictors.py`
    images, labels, bin_classes = load_prep_data(
        dataset_id,
        data_args,
        label_src,
        parts,
    )
    del images, bin_classes

    # Find MLE of every hypothesis distribution
    distrib_mle = {}

    for i, (distrib_id, init_params) in enumerate(distrib_args.items()):
        logging.info(
            'Hypothesis Distribution %d/%d : %s',
            i,
            len(distrib_args),
            distrib_id,
        )

        if distrib_id == 'discrete_uniform':
            # NOTE assumes that mle_adam is used and is returning the negative mle
            distrib_mle[distrib_id] = [distribution_tests.MLEResults(
                -(np.log(1.0 / (init_params['high'] - init_params['low'] + 1)) * len(labels)),
                init_params,
            )]
        elif distrib_id == 'continuous_uniform':
            distrib_mle[distrib_id] = [distribution_tests.MLEResults(
                -(np.log(1.0 / (init_params['high'] - init_params['low'])) * len(labels)),
                init_params,
            )]
        else:
            distrib_mle[distrib_id] = distribution_tests.mle_adam(
                distrib_id,
                labels,
                init_params,
                **mle_args,
            )

        # calculate the different information criterions
        # TODO Loop through top_likelihoods, save BIC
        if isinstance(distrib_mle[distrib_id], list):
            for mle_results in distrib_mle[distrib_id]:
                mle_results.info_criterion = calc_info_criterion(
                    -mle_results.neg_log_likelihood,
                    np.hstack(mle_results.params.values()),
                    info_criterions,
                    len(labels)
                )
        elif isinstance(distrib_mle[distrib_id], distribution_tests.MLEResult):
            # TODO, distribution_tests.mle_adam() returns list always.
            distrib_mle[distrib_id].info_criterion = calc_info_criterion(
                    -distrib_mle[distrib_id].neg_log_likelihood,
                    np.hstack(distrib_mle[distrib_id].params.values()),
                    info_criterions,
                    len(labels)
            )
        else:
            raise TypeError(
                '`distrib_mle` is expected to be either of type list or '
                + 'distribution_tests.MLEResult, instead was {type(distrib_mle)}'
            )

    return distrib_mle


def calc_info_criterion(mle, params, criterions, num_samples=None):
    """Calculate information criterions with mle_list and other information."""
    info_criterion = {}

    if 'bic' in criterions:
        info_criterion['bic'] = distribution_tests.bic(mle, len(params), num_samples)

    if 'aic' in criterions:
        info_criterion['aic'] = distribution_tests.aic(mle, len(params))

    if 'hqc' in criterions:
        info_criterion['hqc'] = distribution_tests.hqc(mle, len(params), num_samples)

    return info_criterion


if __name__ == '__main__':
    #args, data_args, model_args, kfold_cv_args, random_seeds = experiment.io.parse_args('mle')
    args, random_seeds = experiment.io.parse_args('mle')
    # TODO 2nd repeat for focus fold of k folds: load model of that split
    # split data based on specified random_seed

    # TODO Create the distributions and their args to be tested (hard coded options)
    if args.dataset_id == 'LabelMe':
        if args.label_src == 'annotations':
            raise NotImplementedError('`label_src` as "annotations" results in '
                + 'a distribution of distributions of distributions. This '
                + 'needs addressed.')
        else:
            # this is a distribution of distributions, all
            distrib_args = {
                'discrete_uniform': {'high': 7, 'low': 0},
                'continuous_uniform': {'high': 7, 'low': 0},
                'dirichlet_multinomial': {
                    # 3 is max labels per sample, 8 classes
                    'total_count': 3,
                    # w/o prior knowledge, must use all ones
                    'concentration': np.ones(8),
                    #'concentration': np.ones(8) * (1000 / 8),
                },
            }
    elif args.dataset_id == 'FacialBeauty':
        # TODO need to make a distrib of normal distribs, uniforms are fine though.
        distrib_args = {
            'discrete_uniform': {'high': 5, 'low': 1},
            'continuous_uniform': {'high': 5, 'low': 1},
            'dirichlet_multinomial': {
                # 60 is max labels per sample, 5 classes
                'total_count': 60,
                # w/o prior knowledge, must use all ones
                'concentration': np.ones(5),
            },
            #'normal': {'loc': , 'scale':},
        }
    else:
        raise NotImplementedError('The `dataset_id` {args.dataset_id} is not supported.')

    results = test_human_distrib(
        args.dataset_id,
        vars(args.data),
        distrib_args,
        args.label_src,
        args.model.parts,
        mle_args=vars(args.mle),
        info_criterions=['bic'],
        random_seed=None,
    )

    # TODO Save the results
    results = {k: [vars(mle_r) for mle_r in v] for k, v in results.items()}

    experiment.io.save_json(
        os.path.join(args.output_dir, 'mle_results.json'),
        results,
        overwrite=True,
    )

    # TODO if want to see how well the model fits the data, do K Fold Cross validation.
