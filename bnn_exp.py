import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
from experiment.visuals import visualization_snippets
import sjd_log_prob_exp
import json
import tensorflow as tf
import tensorflow_probability as tfp
from psych_metric.distrib import bnn_transform

import src_candidates
from experiment import io

def bnn_mcmc(bnn_args):
    rdm = src_candidates.get_src_sjd('tight_dir_small_mvn', 4)
    s = rdm.sample(10)
    ss0= (s[0] - [1,0,0,0]) @ rdm.transform_matrix.T
    ss1= (s[1] - [1,0,0,0]) @ rdm.transform_matrix.T
    bnn_args={'output_activation': None, 'num_layers':1, 'num_hidden': 100}
    tf_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, ss0.shape[1]],
            name='input_label',
        )

    bnn_out, tf_vars = bnn_transform.bnn_mlp(tf_input, **bnn_args)
    tf_labels = tf.placeholder(
            dtype=tf.float32,
            shape=[None, ss1.shape[1]],
            name='output_labels',
        )
    feed_dict = {
            tf_input: ss0,
            tf_labels: ss1,
        }
    cpu_config = io.get_tf_config(16)
    gpu_config = io.get_tf_config(16, gpus=1)
    weights, badir = bnn_transform.bnn_adam(bnn_out, tf_vars, tf_labels, feed_dict, gpu_config, epochs=15)

    results_dict, feed_dict = bnn_transform.get_bnn_transform(
        ss0,
        ss1,
        kernel_args={'scale':1e-3},
        num_samples=int(4e6),
        burnin=0,
        lag=5,
        bnn_args=bnn_args,
        tf_vars_init=weights,
    )

    iter_results = bnn_transform.bnn_mlp_run_sess(results_dict, feed_dict, sess_config=cpu_config)


def add_bnn_transform_args():
    """Adds the test SJD arguments to the argparser."""
    bnn = parser.add_argument_group(
        'bnn',
        'Arguments pertaining to the Basyesian Neural Network',
    )



    bnn.add_argument(
        '--target_distrib',
        type=multi_typed_arg(
            str,
            json.loads,
        ),
        help=' '.join([
            'Either a str identifer of a distribution or a dict with',
            '"distirb_id" as a key and the parameters of that distribution',
            'that serves as the target distribution.',
        ]),
        dest='sjd.target_distrib',
    )


if __name__ == '__main__':
    args = parse_args()

    bnn_mcmc(

    )
