"""Runs the measures on the raw data (no modeling of distributions).

Notes
-----
Experiement 2 for residuals when target is given target label to predictor:
    Performing this where the target is the actual target label of the task and
    pred is the estimated predictions of the predictor via the BNN MCMC
    generates the distribution of residuals, which is a distribution of a
    measure and part of experiment 2.
"""
from sklearn.metrics import roc_auc_score

from psych_metric.metrics import measure

from experiment import io
from experiment.research.bnn.bnn_mcmc_fwd import load_bnn_io_json
from experiment.research.measure import kldiv

def add_custom_args(parser):
    kldiv.add_custom_args(parser)

    # add other args
    parser.add_argument(
        '--measure',
        default='euclid_dist',
        choices=['euclid_dist', 'kldiv', 'roc_auc', 'all'],
        help=' '.join([
            'Pass if target is task\'s target labels (Exp 2 where measure is',
            'residuals), rather than where the predictor\'s predictions are',
            'the target (Exp 1).',
        ])
    )

    parser.add_argument(
        '--multi_class',
        default='ovo',
        choices=['ovr', 'ovo'],
        help='multi_class for sklearn.metrics.roc_auc_score()',
    )


if __name__ == "__main__":
    # Create argparser
    args = io.parse_args(
        ['sjd'],
        custom_args=add_custom_args,
        description=' '.join([
            'Runs measure on raw data, as one does normally without the',
            'proposed framework.',
        ]),
    )

    #output_dir = io.create_dirs(args.output_dir)
    output_dir = args.output_dir

    givens, conds = load_bnn_io_json(args.data.dataset_filepath)

    # Perform the measurement
    if args.measure == 'all' or args.measure == 'euclid_dist':
        measurements = kldiv.get_l2dists(givens, conds, args.normalize)

        kldiv.save_measures(
            output_dir,
            args.measurement,
            measurements,
            args.quantiles_frac,
            save_raw=not args.do_not_save_raw,
            axis=0,
        )

    if args.measure == 'all' or args.measure == 'kldiv':
        measurements = measure.measure(measure.kldiv_probs, givens, conds)

        kldiv.save_measures(
            output_dir,
            args.measurement,
            measurements,
            args.quantiles_frac,
            save_raw=not args.do_not_save_raw,
            axis=0,
        )

    if args.measure == 'all' or args.measure == 'roc_auc':
        measurements = roc_auc_score(
            givens,
            conds,
            multi_class=args.multi_class,
        )

        # Save the ROC AUC
        io.save_json(
            {'roc_auc': measurements},
            os.path.join(output_dir, 'roc_auc.json'),
        )
