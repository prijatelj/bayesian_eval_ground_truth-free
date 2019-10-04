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


if __name__ == '__main__':
    args = experiment.ioparse_args()

    # TODO First, handle distrib test of src annotations
    # Load the src data, reformat as was done in training in `predictors.py`

    # Find MLE of every hypothesis distribution

    # calculate the different information criterions (Bayes Factor approxed by BIC)

    # TODO 2nd repeat for focus fold of k folds: load model of that split
    # split data based on specified random_seed
