"""General functions used in experiments 1, 2, and 3."""
import os

import h5py
import numpy as np

from psych_metric.metrics.measure import measure

from experiment import io
from experiment.research.bnn import proto_bnn_mcmc

# TODO load bnn in experiment/research/bnn

# Summary func
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


def summary_dict(measurements, quantile_set=None, axis=1):
    """Returns the summaries of different statistics of the measurements. By default, expects the measurements to be of shape

        [samples, num_measurements_per_sample]
    """
    return {
        'overview': summary_arr(measurements, quantile_set=quantile_set),
        'summary_of_means': summary_arr(
            np.mean(measurements, axis=axis),
            quantile_set,
        ),
        'summary_of_maxs': summary_arr(
            np.max(measurements, axis=axis),
            quantile_set,
        ),
        'summary_of_mins': summary_arr(
            np.min(measurements, axis=axis),
            quantile_set,
        ),
        'summary_of_medians': summary_arr(
            np.median(measurements, axis=axis),
            quantile_set,
        ),
    }


def save_raw_measurements(output_dir, measurements, measure_id):
    """Saves raw measures as csv or hdf5 file."""
    if len(measurements.shape) <= 2:
        np.savetxt(
            os.path.join(output_dir, f'{measure_id}.csv'),
            measurements,
            delimiter=',',
        )
    else:
        pass
        # TODO save h5py files


def save_measures(
    output_dir,
    measure_id,
    measurements,
    quantiles_frac=None,
    save_raw=True,
):
    """Convenience function to save measurement output and summarize."""
    if save_raw:
        save_raw_measurements(output_dir, measurements, measure_id)

    if quantiles_frac > 2:
        quantile_set = np.arange(1 + quantiles_frac) / quantiles_frac
    else:
        quantile_set = None

    # Save the summary of the euclidean distances
    io.save_json(
        os.path.join(output_dir, f'{measure_id}_summary.json'),
        summary_dict(measurements, quantile_set, axis=1),
    )

    # Save the flattening of the conditionals via different summarization methods
    io.save_json(
        os.path.join(output_dir, f'{measure_id}_target_samples_flat.json'),
        summary_arr(measurements, quantile_set, axis=1),
    )


def get_l2dists(targets, preds, prob_simplex_normalize=False, axis=2):
    """Obtains euclidean distances from sampling of the predictions."""
    for target_idx in range(len(targets)):
        # NOTE assumes shape of [targets, conditionals, classes]
        preds[target_idx] = targets[target_idx] - preds[target_idx]
    differences = preds

    if prob_simplex_normalize:
        # Normalizes by the largest possible distance within the probability
        # simplex, which is the distance form one vertex to any other vertex
        # because the probability simplex is regular (ie. 2-simplex is a
        # equilateral triangle).
        return np.linalg.norm(differences, axis=axis) / np.sqrt(2)

    return np.linalg.norm(differences, axis=axis)


def add_custom_args(parser):
    proto_bnn_mcmc.add_custom_args(parser)

    # add other args
    parser.add_argument(
        '--target_is_task_target',
        action='store_true',
        help=' '.join([
            'Pass if target is task\'s target labels (Exp 2 where measure is',
            'residuals), rather than where the predictor\'s predictions are',
            'the target (Exp 1).',
        ])
    )

    """
    parser.add_argument(
        '--loaded_bnn_outputs',
        default=None,
        action='store_true',
        help=' '.join([
            'Pass if the bnn_weights file is actually the bnn output, rather',
            'than the expected BNN weight sets. NOT IMPLEMNETED yet.',
        ])
    )
    """

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

    parser.add_argument(
        '--do_not_save_raw',
        action='store_true',
        help='Pass if not to save the raw measurements.'
    )


if __name__ == '__main__':
    # TODO create argparser

    # TODO Load the data

    # Do exp 1 (euclidean dists) or exp 2 using using KLDiv, euclid dist
    # (residuals) or confusion matrix

    # TODO if euclidean measure call other scripts and return results

    # TODO save results
    pass
