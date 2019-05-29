"""
The baseline metrics used to compare two (or multiple) random variables to one
another, providing an (ideally) informative and interpretable distance.

To be mathematically correct, some of these are measures, that provide the size
or quantity of a set, rather than a distance (such as Mutual Information)
"""
import itertools
from math import sqrt

import numpy as np
from scipy.stats import multinomial, entropy, wasserstein_distance
from sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt

#from psych_metric.metrics.base_metric import BaseMetric

#class BaselineMetrics(BaseMetric):
#    def __init__(self):
#        pass

def load(input_dir, summary_csv='summary.csv', annotation_aggregation_csv='annoatation_aggregation.csv', result_filename='metric_analysis_ground_truth_vs_aggregation.csv'):
    """Traverses the given root directory to find directories that contain a
    summary.csv, loads the dataset with ground truth based on dataset_id, and
    compares that ground truth to the contents within the annotation
    aggregation. The calculated metrics are then saved in the same

    Parameters
    ----------
    input_dir : str
        The filepath to the input directory to be traversed to find directories
        that contain the results of annotation aggregation methods.
    summary : str, optional
        The filename of the summary csv
    annotation_aggregation : str, optional
        The filename of the annotation aggregation output.
    result_filename : str, optional
        The filename of the file where the metric calculations are stored.
        Relative to the directory containing the summary and annotation
        aggregation files. If None, then it simply returns

    Yields
    ------
    dataset, annotation_aggregation, parent_dir
    """

    yield

def calculate(target, predictions, task_type=None, metrics=None):
    """Function handler that calls the appropriate metrics on the given data.

    Parameters
    ----------
    target : array-like
        The target values the predictions are compared to.
    predictions : array-like
        The prediction values to be compared to the target values.
    task_type : str, optional
        The task type performed on the data. If not provided, only task type
        agnostic metrics will be used, if any apply.

    metrics : set(str), optional
        A set of metric identifiers to be calculated if appropriate to the data
        and task type. By default, attempts to calcualte all possible metrics.

    Returns
    -------
        dict or dataframe of the metrics and their calculations on the data.
    """
    metric_results = dict()

    # TODO need to check dimensions of target and predictions and include in conditionals.

    # TODO could make this way more efficient since there is a lot of orverlapping parts among these metrics.

    # task_type agnostic metrics:
    if metrics is None or 'mutual_information' in metrics or 'kl_divergence' in metrics:
        # NOTE confirm mutual information == kl_divergence
        metric_results['mutual_information'] = entropy(target, predictions)

    #if metrics is None or 'variational_information' in metrics:
    #    # the metric version of mutual information.

    if metrics is None or 'wasserstein' in metrics or 'earth_movers_distance' in metrics:
        metric_results['wasserstein'] = wasserstein_distance(target, predictions)

    if task_type == 'regression':
        if metrics is None or'mean_absolute_error' in metrics:
            metric_results['mean_absolute_error'] = skl_metrics.mean_absolute_error(target, predictions)
        if metrics is None or 'mse' in metrics or 'mean_squared_error' in metrics:
            metric_results['mean_squared_error'] = skl_metrics.mean_squared_error(target, predictions)

        if metrics is None or 'rmse' in metrics:
            metric_results['root_mean_squared_error'] = math.sqrt(skl_metrics.mean_squared_error(target, predictions))

        if metrics is None or'median_absolute_error' in metrics:
            metric_results['median_absolute_error'] = skl_metrics.median_absolute_error(target, predictions)

        if metrics is None or 'r_squared' in metrics:
            metric_results['r_squared'] = skl_metrics.r2_score(target, predictions)

    # TODO require standardized format of the predictions and target to be able to infer the correct metric to use (ie. a sample is a scalar to scalar comparison, or distribution to distribution, etc.)


    elif 'classification' in task_type:
        if metrics is None or 'confusion_matrix' in metrics:
            if 'binary' in task_type
                metric_results['confusion_matrix'] = skl_metrics.confusion_matrix(target, predictions)
            else:

                metric_results['confusion_matrix'] = skl_metrics.multilabel_confusion_matrix(target, predictions)

        #if metrics is None or 'specificity' in metrics:
        #    metric_results['specificity'] = skl_metrics.specificity(target, predictions)

        #if metrics is None or 'sensitivity' in metrics:
        #    metric_results['sensitivity'] = skl_metrics.sensitivity(target, predictions)

        if metrics is None or 'accuracy' in metrics:
            metric_results['accuracy'] = skl_metrics.accuracy_score(target, predictions)

        if metrics is None or 'precision' in metrics:
            metric_results['precision'] = skl_metrics.precision_score(target, predictions)

        if metrics is None or 'recall' in metrics:
            metric_results['recall'] = skl_metrics.recall_score(target, predictions)

        if metrics is None or 'f1' in metrics:
            metric_results['f1'] = skl_metrics.f1_score(target, predictions)

        #if metrics is None or 'roc' in metrics: # area under curve
        #    #metric_results['accuracy'] = skl_metrics.recall_score(target, predictions)
        #    # TODO use the roc class.

        #if metrics is None or 'toc' in metrics: # area under curve


    #elif 'ordering' in task_type: # ranking

    #elif 'mapping' in task_type:

    return metric_results


def parse_args():

    return args


if __name__ == '__main__':
    args = parse_args()

    for dataset, annotation_aggregation, parent_dir in load(args.input_dir, args.summary_csv, args.annotation_aggregation_csv):
        metric_results = calculate(dataset['ground_truth'], annotation_aggregation, dataset.task_type)
        metric_results.to_csv(os.path.join(parent_dir, args.result_filename), index=False)
