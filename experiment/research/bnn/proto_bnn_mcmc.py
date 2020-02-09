"""Prototyping bnn mcmc on CRC."""
import logging
import os
import json
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib import bnn_transform

from experiment.research.bnn import bnn_exp
from experiment import io
from experiment.research.sjd import src_candidates


def mcmc_sample_log_prob(params,data,targets,scale_identity_multiplier=0.01):
    bnn_data = tf.convert_to_tensor(data.astype(np.float32),dtype=tf.float32)
    bnn_target = tf.convert_to_tensor(targets.astype(np.float32),dtype=tf.float32)

    hidden_weights, hidden_bias, output_weights, output_bias = params
    hidden = tf.nn.sigmoid(bnn_data @ hidden_weights + hidden_bias)

    output = hidden @ output_weights + output_bias

    return tf.reduce_sum(
        tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros([data.shape[1]]),
            scale_identity_multiplier=scale_identity_multiplier,
        ).log_prob(output-bnn_target)
    )


def setup_rwm_sim(
    width=10,
    sample_size=10,
    scale_identity_multiplier=0.01,
    dim=4,
):
    rdm = src_candidates.get_src_sjd('tight_dir_small_mvn', dim)
    s = rdm.sample(sample_size)
    data = rdm.transform_distrib.simplex_transform.to(s[0])
    targets = rdm.transform_distrib.simplex_transform.to(s[1])

    init_state = [
        np.random.normal(scale=12**0.5 , size=(dim,width)).astype(np.float32),
        np.zeros([width], dtype=np.float32),
        np.random.normal(scale=0.48**0.5 , size=(width,dim)).astype(np.float32),
        np.zeros([dim], dtype=np.float32)]

    sample_log_prob = partial(
        bnn_transform.mcmc_sample_log_prob,
        origin_adjust=rdm.transform_distrib.simplex_transform.origin_adjust,
        rotation_mat=rdm.transform_distrib.simplex_transform.change_of_basis_matrix,
        scale_identity_multiplier=scale_identity_multiplier,
    )

    #return data, targets, sample_log_prob, init_state
    return data, targets, sample_log_prob, init_state


def adam_init(data, targets, width, dim, epochs, cpus=1, cpu_cores=16, gpus=0, init_vars=None):
    config = io.get_tf_config(cpus, cpu_cores, gpus)

    tf_in = tf.placeholder(dtype=tf.float32, shape=[None, data.shape[1]])
    tf_out = tf.placeholder(dtype=tf.float32, shape=[None, data.shape[1]])

    feed_dict = {tf_in: data, tf_out: targets}

    bnn_out, tf_vars = bnn_transform.bnn_mlp(
        tf_in,
        1,
        width,
        output_activation=None,
        output_use_bias=True,
    )

    if init_vars is not None:
        init_vars = {tf_vars[i]:init_vars[i] for i in range(len(init_vars))}

    new_state, iter_results = bnn_transform.bnn_adam(
        bnn_out,
        tf_vars,
        tf_out,
        feed_dict,
        epochs=epochs,
        tf_config=config,
        init_vars=init_vars,
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
    parallel=1,
):
    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        new_state_fn = tfp.mcmc.random_walk_normal_fn(scale=rwm_scale),
    )

    return sample_chain_run(
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
        parallel=parallel,
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
    step_adjust_id='Simple',
    config=None,
    parallel=1,
):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )
    if num_adaptation_steps:
        if step_adjust_id == 'Simple':
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                kernel,
                num_adaptation_steps=num_adaptation_steps,
            )
        else:
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                kernel,
                num_adaptation_steps=num_adaptation_steps,
            )

    return sample_chain_run(
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
        parallel=parallel,
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
    num_adaptation_steps=0,
    step_adjust_id='Simple',
    config=None,
    parallel=1,
):
    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        step_size=step_size,
    )

    if num_adaptation_steps > 0:
        # TODO setup NUTS to use adaptative step sizes
        trans_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=kernel,
            bijector=[tfp.bijectors.Identity()] * len(init_state),
        )

        if step_adjust_id == 'Simple':
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=trans_kernel,
                num_adaptation_steps=num_adaptation_steps,
                target_accept_prob=0.75,
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    inner_results=pkr.inner_results._replace(step_size=new_step_size)
                ),
                step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
            )
        else:
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=trans_kernel,
                num_adaptation_steps=num_adaptation_steps,
                target_accept_prob=0.75,
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    inner_results=pkr.inner_results._replace(step_size=new_step_size)
                ),
                step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
            )

    return sample_chain_run(
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
        parallel=parallel,
    )


def sample_chain_run(
    init_state,
    kernel,
    num_results=10000,
    burnin=0,
    lag=0,
    config=None,
    parallel=1,
):
    if parallel <= 1:
        samples, trace = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=init_state,
            kernel=kernel,
            num_burnin_steps=burnin,
            num_steps_between_results=lag,
            parallel_iterations=1,
        )
        with tf.Session(config=config) as sess:
            sess.run((
                tf.global_variables_initializer(),
                tf.local_variables_initializer(),
            ))

            output = sess.run([samples,trace])
            new_starting_state = [x[-1] for x in output[0]]

        return output, new_starting_state
    else:
        # NOTE parallel chains saves only is_accepted for trace_fn
        # create multiple of the same chains, w/ diff seeds, gets samples fast
        chains_results = []
        for i in range(parallel):
            chains_results.append(tfp.mcmc.sample_chain(
                num_results=num_results,
                current_state=init_state,
                kernel=kernel,
                num_burnin_steps=burnin,
                num_steps_between_results=lag,
                parallel_iterations=1,
                trace_fn=lambda current_state, kernel_results: kernel_results.is_accepted,
            ))

        with tf.Session(config=config) as sess:
            sess.run((
                tf.global_variables_initializer(),
                tf.local_variables_initializer(),
            ))
            output = sess.run(chains_results)

        return output


def add_custom_args(parser):
    bnn_exp.add_custom_args(parser)

    parser.add_argument(
        '--num_samples',
        default=10,
        type=int,
        help='The number of src simulation samples.',
    )

    parser.add_argument(
        '--dim',
        default=3,
        type=int,
        help='The number of dimensions of the discrete distribution data (input and output).',
    )

    parser.add_argument(
        '--parallel_chains',
        default=1,
        type=int,
        help='The number of chains to be run in parallel for sampling.',
    )

    parser.add_argument(
        '--bnn_weights_file',
        default=None,
        help='Path to the bnn weights file.',
    )

    #parser.add_argument(
    #    '--no_visuals',
    #    default=None,
    #    help='Does not save visuals.',
    #)

    parser.add_argument(
        '--',
        default=1,
        type=int,
        help='The number of chains to be run in parallel for sampling.',
    )


if __name__ == '__main__':
    args = io.parse_args(custom_args=add_custom_args)

    # Create directory
    output_dir = args.output_dir
    output_dir = io.create_dirs(output_dir)
    logging.info('Created the output directories')

    if os.path.isfile(args.data.dataset_filepath):
        with open(args.data.dataset_filepath, 'r') as f:
            data = json.load(f)
            givens = np.array(data['givens'], dtype=np.float32)
            conditionals = np.array(data['conditionals'], dtype=np.float32)
            change_of_basis = np.array(data['change_of_basis'], dtype=np.float32)

        scale_identity_multiplier=args.mcmc.scale_identity_multiplier,
        sample_log_prob = partial(
            bnn_transform.mcmc_sample_log_prob,
            scale_identity_multiplier=args.mcmc.scale_identity_multiplier,
        )

        if not isinstance(args.bnn_weights_file, str):
            raise TypeError('bnn_weights_file must be provided when dataset_filepath is given')

        with open(args.bnn_weights_file, 'r') as f:
            init_state = json.load(f)
            init_state = [np.array(x, dtype=np.float32) for x in init_state]
    else:
        givens, conditionals, sample_log_prob, init_state = setup_rwm_sim(
            width=args.bnn.num_hidden,
            sample_size=args.num_samples,
            scale_identity_multiplier=args.mcmc.scale_identity_multiplier,
            dim=args.dim,
        )

        io.save_json(
            os.path.join(output_dir, 'data_for_bnn.json'),
            {'givens': givens, 'conditionals': conditionals},
        )

    logging.info('Setup the simulation data and the log prob function')

    if args.adam_epochs > 0:
        logging.info('Starting ADAM initialization training')
        if args.bnn_weights_file:
            init_vars = init_state
        init_state, loss = adam_init(
            givens,
            conditionals,
            args.bnn.num_hidden,
            givens.shape[1],
            args.adam_epochs,
            cpus=args.cpu,
            cpu_cores=args.cpu_cores,
            gpus=args.gpu,
            init_vars=init_vars
        )
        logging.info('Finished ADAM initialization training')

    config = io.get_tf_config(args.cpu, args.cpu_cores, 0)

    logging.info('Starting MCMC training')

    if args.mcmc.kernel_id == 'RandomWalkMetropolis':
        output, new_starting_state = run_rwm(
            givens,
            conditionals,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            rwm_scale=args.mcmc.kernel.step_size,
            config=config,
        )
        logging.info('Finished RandomWalkMetropolis')

        accept_total = output[1].is_accepted.sum()
        accept_rate = output[1].is_accepted.mean()

        acf_log_prob = acf(
            output[1].accepted_results.target_log_prob,
            nlags=int(args.mcmc.sample_chain.num_results / 4),
        )

        final_step_size = args.mcmc.kernel.step_size

        logging.info('Starting RandomWalkMetropolis specific visuals')
        plt.plot(output[1].accepted_results.target_log_prob)
        plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
        plt.close()

        pd.DataFrame(acf_log_prob).plot(kind='bar')
        plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
        plt.close()

    elif args.mcmc.kernel_id == 'HamiltonianMonteCarlo':
        output = run_hmc(
            givens,
            conditionals,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            step_size=args.mcmc.kernel.step_size, # 5e-4
            num_leapfrog_steps=args.mcmc.kernel.num_leapfrog_steps, # 5
            num_adaptation_steps=args.mcmc.num_adaptation_steps,
            config=config,
            step_adjust_id=args.mcmc.step_adjust_id,
            parallel=args.parallel_chains,
        )
        logging.info('Finished HamiltonianMonteCarlo')

        if args.parallel_chains <= 1:
            output, new_starting_state = output

            if args.mcmc.num_adaptation_steps > 0:
                mcmc_results = output[1].inner_results
                final_step_size = output[1].new_step_size[-1]
            else:
                mcmc_results = output[1]
                final_step_size = args.mcmc.kernel.step_size

            accept_total = mcmc_results.is_accepted.sum()
            accept_rate = mcmc_results.is_accepted.mean()

            acf_log_prob = acf(
                mcmc_results.accepted_results.target_log_prob,
                nlags=int(args.mcmc.sample_chain.num_results / 4),
            )

            logging.info('Starting HamiltonianMonteCarlo specific visuals')
            plt.plot(mcmc_results.accepted_results.target_log_prob)
            plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
            plt.close()

            pd.DataFrame(acf_log_prob).plot(kind='bar')
            plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
            plt.close()
        else:
            sampled_weights = {}
            for i in range(0, len(output)):
                sampled_weights[i] = {}
                sampled_weights[i]['weights'] = output[i].all_states
                # the trace is only boolean is_accepted
                sampled_weights[i]['is_accepted'] = output[i].trace

            io.save_json(
                os.path.join(output_dir, 'sampled_weights.json'),
                sampled_weights,
            )

    elif args.mcmc.kernel_id == 'NoUTurnSampler':
        output, new_starting_state = run_nuts(
            givens,
            conditionals,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            step_size=args.mcmc.kernel.step_size, # 5e-4
            config=config,
            num_adaptation_steps=args.mcmc.num_adaptation_steps,
            step_adjust_id=args.mcmc.step_adjust_id,
        )
        logging.info('Finished NoUTurnSampler')

        if args.mcmc.num_adaptation_steps > 0:
            # Need to extract from Adaptive step and from Transformed kernels
            mcmc_results = output[1].inner_results.inner_results
            final_step_size = output[1].new_step_size[-1]
        else:
            mcmc_results = output[1]
            final_step_size = args.mcmc.kernel.step_size

        accept_total = mcmc_results.is_accepted.sum()
        accept_rate = mcmc_results.is_accepted.mean()

        acf_log_prob = acf(
            mcmc_results.target_log_prob,
            nlags=int(args.mcmc.sample_chain.num_results / 4),
        )

        logging.info('Starting NoUTurnSampler specific visuals')
        plt.plot(mcmc_results.target_log_prob)
        plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
        plt.close()

        pd.DataFrame(acf_log_prob).plot(kind='bar')
        plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
        plt.close()

    logging.info('Finished MCMC training and specific kernel givens saving.')


    if args.parallel_chains <= 1:
        io.save_json(
            os.path.join(output_dir, 'last_weights.json'),
            new_starting_state,
        )

        acf_lag = {
            'accept_total': accept_total,
            'accept_rate': accept_rate,
            '0.5': np.where(np.abs(acf_log_prob) < 0.5)[0][:10],
            '0.1': np.where(np.abs(acf_log_prob) < 0.1)[0][:10],
            '0.01': np.where(np.abs(acf_log_prob) < 0.01)[0][:10],
            'final_step_size': final_step_size,
        }
        io.save_json(
            os.path.join(output_dir, 'acf_lag.json'),
            acf_lag,
        )
