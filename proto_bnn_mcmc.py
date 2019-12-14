"""Prototyping bnn mcmc on CRC."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import tensorflow as tf
import tensorflow_probability as tfp

import bnn_exp
from experiment import io
from psych_metric.distrib import bnn_transform
import src_candidates


def setup_rwm_sim(
    width=10,
    sample_size=10,
    scale_identity_multiplier=0.01,
):
    rdm = src_candidates.get_src_sjd('tight_dir_small_mvn', 4)
    s = rdm.sample(sample_size)
    data= (s[0] - [1, 0, 0, 0]) @ rdm.transform_matrix.T
    targets= (s[1] - [1, 0, 0, 0]) @ rdm.transform_matrix.T

    dim = data.shape[1]

    def sample_log_prob(params,data,targets,scale_identity_multiplier=0.01):
        bnn_data = tf.convert_to_tensor(data.astype(np.float32),dtype=tf.float32)
        bnn_target = tf.convert_to_tensor(targets.astype(np.float32),dtype=tf.float32)
        hidden_weights, hidden_bias, output_weights, output_bias = params
        hidden = tf.nn.sigmoid(bnn_data @ hidden_weights + hidden_bias)
        output = hidden @ output_weights + output_bias
        return tf.reduce_sum(tfp.distributions.MultivariateNormalDiag(loc=tf.zeros([dim]),scale_identity_multiplier=scale_identity_multiplier).log_prob(output-targets))

    init_state = [
        np.random.normal(scale=12**0.5 , size=(dim,width)).astype(np.float32),
        np.zeros([width], dtype=np.float32),
        np.random.normal(scale=0.48**0.5 , size=(width,dim)).astype(np.float32),
        np.zeros([dim], dtype=np.float32)]

    return data, targets, sample_log_prob, init_state


def adam_init(data, targets, width, dim, epochs, cpus=1, cpu_cores=16, gpus=0):
    config = io.get_tf_config(cpus, cpu_cores, gpus)

    tf_in = tf.placeholder(dtype=tf.float32, shape=[None, data.shape[1]])
    tf_out = tf.placeholder(dtype=tf.float32, shape=[None, data.shape[1]])

    feed_dict = {tf_in:data, tf_out:targets}

    bnn_out, tf_vars = bnn_transform.bnn_mlp(
        tf_in,
        1,
        width,
        output_activation=None,
        output_use_bias=True,
    )
    new_state, iter_results = bnn_transform.bnn_adam(
        bnn_out,
        tf_vars,
        tf_out,
        feed_dict,
        epochs=epochs,
    )

    return new_state, iter_results['loss']


def run_rwm(
    data,
    targets,
    sample_log_prob,
    init_state,
    num_results=10000,
    burnin=0,
    lag=0,
    rwm_scale=2e-4,
    config=None,
):
    kernel=tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        new_state_fn = tfp.mcmc.random_walk_normal_fn(scale=rwm_scale),
    )

    return sample_chain_run(
        data,
        targets,
        sample_log_prob,
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
    )


def run_hmc(
    data,
    targets,
    sample_log_prob,
    init_state,
    num_results=10000,
    burnin=0,
    lag=0,
    step_size=5e-4,
    num_leapfrog_steps=5,
    num_adaptation_steps=None,
    config=None,
):
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )
    if num_adaptation_steps:
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            kernel,
            num_adaptation_steps=num_adaptation_steps,
        )

    return sample_chain_run(
        data,
        targets,
        sample_log_prob,
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
    )


def run_nuts(
    data,
    targets,
    sample_log_prob,
    init_state,
    num_results=10000,
    burnin=0,
    lag=0,
    step_size=5e-4,
    config=None,
):
    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        step_size=step_size,
    )

    return sample_chain_run(
        data,
        targets,
        sample_log_prob,
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
    )


def sample_chain_run(
    data,
    targets,
    sample_log_prob,
    init_state,
    kernel,
    num_results=10000,
    burnin=0,
    lag=0,
    config=None,
):
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=kernel,
        num_burnin_steps=burnin,
        num_steps_between_results=lag,
        parallel_iterations=1,
    )

    with tf.Session(config=config) as sess:
        output = sess.run([samples,trace])
        new_starting_state = [x[-1] for x in output[0]]

    return output, new_starting_state


def add_custom_args(parser):
    bnn_exp.add_custom_args(parser)

    parser.add_argument(
        '--num_samples',
        default=10,
        type=int,
        help='The number of src simulation samples.',
    )


if __name__ == '__main__':
    args = io.parse_args(custom_args=add_custom_args)

    # Create directory
    output_dir = args.data.dataset_filepath
    output_dir = io.create_dirs(output_dir)


    data, targets, sample_log_prob, init_state = setup_rwm_sim(
        width=args.bnn.num_hidden,
        sample_size=args.num_samples,
        scale_identity_multiplier=args.mcmc.diff_scale,
    )

    if args.adam_epochs > 0:
        init_state, loss = adam_init(
            data,
            targets,
            args.bnn.num_hidden,
            data.shape[1],
            args.adam_epochs,
            cpus=args.cpu,
            cpu_cores=args.cpu_cores,
            gpus=args.gpu,
        )

    config = io.get_tf_config(args.cpu, args.cpu_cores, 0)

    if args.mcmc.kernel_id == 'RandomWalkMetropolis':
        output, new_starting_state = run_rwm(
            data,
            targets,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            rwm_scale=args.mcmc.kernel.step_size,
            config=config,
        )

        acf_lag = {
            '0.5': np.where(np.abs(output[1].accepted_results.target_log_prob) < 0.5)[0][:10],
            '0.1': np.where(np.abs(output[1].accepted_results.target_log_prob) < 0.1)[0][:10],
            '0.01': np.where(np.abs(output[1].accepted_results.target_log_prob) < 0.01)[0][:10],
        }

        plt.plot(output[1].accepted_results.target_log_prob)
        plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
        plt.close()

        pd.DataFrame(acf(output[1].accepted_results.target_log_prob[:int(args.mcmc.sample_chain.num_results / 4)])).plot(kind='bar')
        plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
        plt.close()

    elif args.mcmc.kernel_id == 'HamiltonianMonteCarlo':
        output, new_starting_state = run_hmc(
            data,
            targets,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            step_size=args.mcmc.kernel.step_size, # 5e-4
            num_leapfrog_steps=args.mcmc.kernel.num_leapfrog_steps, # 5
            num_adaptation_steps=args.mcmc.num_adaptation_steps,
            config=config,
        )

        acf_lag = {
            '0.5': np.where(np.abs(output[1].accepted_results.target_log_prob) < 0.5)[0][:10],
            '0.1': np.where(np.abs(output[1].accepted_results.target_log_prob) < 0.1)[0][:10],
            '0.01': np.where(np.abs(output[1].accepted_results.target_log_prob) < 0.01)[0][:10],
        }

        plt.plot(output[1].accepted_results.target_log_prob)
        plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
        plt.close()

        pd.DataFrame(acf(output[1].accepted_results.target_log_prob[:int(args.mcmc.sample_chain.num_results / 4)])).plot(kind='bar')
        plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
        plt.close()

    elif args.mcmc.kernel_id == 'NoUTurnSampler':
        output, new_starting_state = run_nuts(
            data,
            targets,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            step_size=args.mcmc.kernel.step_size, # 5e-4
            config=config,
        )

        acf_lag = {
            '0.5': np.where(np.abs(output[1].target_log_prob) < 0.5)[0][:10],
            '0.1': np.where(np.abs(output[1].target_log_prob) < 0.1)[0][:10],
            '0.01': np.where(np.abs(output[1].target_log_prob) < 0.01)[0][:10],
        }

        plt.plot(output[1].target_log_prob)
        plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
        plt.close()

        pd.DataFrame(acf(output[1].target_log_prob[:int(args.mcmc.sample_chain.num_results / 4)])).plot(kind='bar')
        plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
        plt.close()

    io.save_json(
        os.path.join(output_dir, 'last_weights.json'),
        new_starting_state,
    )

    io.save_json(
        os.path.join(output_dir, 'acf_lag.json'),
        acf_lag,
    )
