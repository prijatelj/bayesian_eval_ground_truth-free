"""
The baseline metrics used to compare two (or multiple) random variables to one
another, providing an (ideally) informative and interpretable distance.

To be mathematically correct, some of these are measures, that provide the size
or quantity of a set, rather than a distance (such as Mutual Information)
"""
import numpy as np
from scipy.stats import multinomial
import itertools
import matplotlib.pyplot as plt
import theano.tensor as tt
import pymc3 as pm

from psych_metric.metrics.base_metric import BaseMetric

class BaselineMetrics(BaseMetric):
    def __init__(self):
        pass
