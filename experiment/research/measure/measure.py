"""General functions used in experiments 1, 2, and 3."""
import os

import h5py
import numpy as np

from psych_metric.metrics.measure import measure

from experiment import io

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

def save_measures(
    output_dir,
    measure_id,
    measurements,
    quantiles_frac=None,
    save_raw=True,
):
    """Convenience function to save measurement output and summarize."""
    if save_raw:
        if len(measurements.shape) <= 2:
            np.savetxt(
                os.path.join(output_dir, f'{measure_id}.csv'),
                measurements,
                delimiter=',',
            )
        else:
            pass
            # TODO save h5py files

    if quantiles_frac > 2:
        quantile_set = np.arange(1 + quantiles_frac) / quantiles_frac
    else:
        quantile_set = None

    # Save the summary of the euclidean distances
    io.save_json(
        os.path.join(output_dir, f'{measure_id}_summary.json'),
        {
            'overview': summary_arr(measurements, quantile_set=quantile_set),
            'summary_of_means': summary_arr(
                np.mean(measurements, axis=1),
                quantile_set,
            ),
            'summary_of_maxs': summary_arr(
                np.max(measurements, axis=1),
                quantile_set,
            ),
            'summary_of_mins': summary_arr(
                np.min(measurements, axis=1),
                quantile_set,
            ),
            'summary_of_medians': summary_arr(
                np.median(measurements, axis=1),
                quantile_set,
            ),
        },
    )

    # Save the flattening of the conditionals via different summarization methods
    io.save_json(
        os.path.join(output_dir, f'{measure_id}_target_samples_flat.json'),
        summary_arr(measurements, quantile_set, axis=1),
    )


if __name__ == '__main__':
    # TODO create argparser

    # TODO Load the data

    # Do exp 1 (euclidean dists) or exp 2 using using KLDiv, euclid dist
    # (residuals) or confusion matrix

    # TODO if euclidean measure call other scripts and return results

    # TODO save results
    pass
