"""Forward pass of the BNN """
import numpy as np

from psych_metric.distrib.bnn.bnn_mcmc import BNN_MCMC

import experiment.io

if __name__ == '__main__':
    pass
    # TODO Load sampled weights
    # TODO combine sampled weights into a list
    # TODO Load dataset's labels (given = target, conditionals = pred)
    # TODO Create instance of BNN_MCMC
    # TODO run KNNDE using BNN_MCMC.predict(givens, weights)
