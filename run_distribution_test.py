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


def test_human_distribs(
    dataset_id,
    data_config,
    distrib_config,
    mle_args=None,
    info_criterions=['bic'],
    random_seed=None,
):
    """Hypothesis testing of distributions of the human annotations.

    Parameters
    ----------
    dataset_id : str
        Identifier of dataset.
    data_config : dict
        Dict containing information pertaining to the data.
    distrib_config : dict
        Dict of str distrib ids to their initial parameters
    random_seed : int, optional
        Integer to use as the random seeds for MLE.
    """

    # TODO First, handle distrib test of src annotations
    # Load the src data, reformat as was done in training in `predictors.py`
    images, labels, bin_classes = load_prep_data(
        dataset_id,
        data_config,
        label_src,
        model_config['parts'],
    )

    # Find MLE of every hypothesis distribution
    distrib_mle : {}

    for i, (distrib_id, init_params) in enumerate(distrib_config.items()):
        logging.info(
            'Hypothesis Distribution %d/%d : %s',
            i,
            len(distrib_config),
            distrib_id,
        )

        distrib_mle[distrib_id] = distribution_tests.mle_adam(
            distrib_id,
            labels,
            init_params,
            **mle_args,
        )

        # calculate the different information criterions (Bayes Factor approxed by BIC)
        # TODO Loop through top_likelihoods, save BIC
        if isinstance(distrib_mle[distrib_id], list):
            for mle_list for distrib_mle[distrib_id]:
                mle_list.append(calc_info_criterion(
                    mle_list[0],
                    mle_list[1],
                    info_criterions,
                    len(labels)
                ))
        else:
            distrib_mle.append(calc_info_criterion(
                    distrib_mle[0],
                    distrib_mle[1],
                    info_criterions,
                    len(labels)
            ))

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
    args, data_config, model_config, kfold_cv_args, random_seeds = experiment.io.parse_args()
    # TODO 2nd repeat for focus fold of k folds: load model of that split
    # split data based on specified random_seed
