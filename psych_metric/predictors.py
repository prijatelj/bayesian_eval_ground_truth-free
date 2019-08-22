"""Simple testing concept."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import math
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

from hierarchical_open_set_recognition import loss
from hierarchical_open_set_recognition import ml_models

# TODO tensorboard


def loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def activation_summary(x):
    # tensor_name = re.sub('_[0-9]*/', '', x.op.name)
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def kfold_cv(model, data, k=5, random_seed=None, save_pred=True, save_model=True, stratified=True, test_focus_fold=True):
    """Generator for kfold cross validation.

    Parameters
    ----------
    model :
        custom class or something needs: train, test, save
    data :
    k : int, optional
    random_seed : int, optional
    save_pred : bool, optional
    save_model : bool, optional
    stratified : bool, optional
    test_focus_fold : bool, optional
        If True (default), the single focus fold will be the current
        iteration's test set while the rest are used for training the model,
        otherwise the focus fold is the only fold used for training and the
        rest are used for testing.
    """

    # TODO data index splitting


    for i, (other_folds, focus_fold) in enumerate(data_folds):
        logging.info(f'{i}/{k} fold cross validation: Training starting')

        model.train(other_folds if test_focus_fold else focus_fold, **train_args)

        logging.info(f'{i}/{k} fold cross validation: Testing starting')
        pred = model.predict(test if test_focus_fold else train, **test_args)

        result = [summary]
        if save_pred:
            result.append(pred)
        if save_model:
            result.append(model)
        yield *result

def run_lenet(loss_id, epochs=1, dense_size=500, loss_weights=1.0, joint_loss_method='arithmetic_mean', batch_size=1000, model_id='lenet++', shuffle_data=True, num_target_labels=None, output_dir=None, random_seed=None, sess_config=None, summary_path='', **kwargs):
    """Runs LeNet[++] on MNIST."""
    summary_params = locals()
    summary_params.pop('sess_config')

    if random_seed:
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    num_classes = len(np.unique(y_train))

    # Binarize the data
    label_bin = LabelBinarizer()
    label_bin.fit(y_train)

    y_train = label_bin.transform(y_train).astype('float32', copy=False)
    y_test = label_bin.transform(y_test).astype('float32', copy=False)

    x_train = x_train[..., np.newaxis].astype('float32', copy=False)
    x_test = x_test[..., np.newaxis].astype('float32', copy=False)

    # tensorflow prep data
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle_data:
        dataset = dataset.shuffle(batch_size, seed=random_seed)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    (features, labels) = iterator.get_next()

    tf.summary.image(features.name, features[:, :, :, 0, tf.newaxis])

    # create model
    if model_id == 'lenet++':
        lenet_output, lenet_fc = ml_models.tf_models.lenet_pp(
            features,
            num_classes,
            return_fc=True,
        )
    else:
        lenet_output, lenet_fc = ml_models.tf_models.lenet(
            features,
            num_classes,
            dense_size,
            return_fc=True,
        )
    tf.summary.histogram(lenet_fc.op.name, lenet_fc)
    tf.summary.histogram(lenet_output.op.name, lenet_output)
    activation_summary(lenet_fc)
    activation_summary(lenet_output)

    # TODO add to loss collection, add losses()?. Add all scalars to losses.
    losses = []
    if 'softmax' in loss_id or loss_id == 'center_loss' or loss_id == 'objectosphere' or loss_id == 'tcl':
        # TODO make it so TCL can be used on its own, ie. not use lenet_outputs
        losses.append(tf.losses.softmax_cross_entropy(labels, lenet_output))
        tf.losses.add_loss(losses[-1])

    if loss_id == 'center_loss' or loss_id == 'cl':
        with tf.variable_scope('center_loss') as scope:
            cl_dist, centers = loss.losses.center_loss(labels, lenet_fc, return_centers=True)
            losses.append(cl_dist)

            tf.summary.image(f'{centers.op.name}', centers[tf.newaxis, :, :, tf.newaxis])

            # activation_summary(centers)
            tf.losses.add_loss(losses[-1])
    elif loss_id == 'triplet_center_loss' or loss_id == 'tcl':
        with tf.variable_scope('triplet_center_loss') as scope:
            tcl_dist, centers = loss.losses.triplet_center_loss(
                labels,
                lenet_fc,
                batch_size,
                0 if 'margin' not in kwargs else kwargs['margin'],
                return_centers=True,
            )
            losses.append(tcl_dist)

            tf.summary.image(f'{centers.op.name}', centers[tf.newaxis, :, :, tf.newaxis])

            # activation_summary(centers)
            tf.losses.add_loss(losses[-1])
    elif loss_id == 'ring_loss' or loss_id == 'objectosphere' and num_target_labels:
        with tf.variable_scope('ring_loss') as scope:
            flags = tf.dtypes.cast(tf.argmax(labels, axis=1) < num_target_labels, tf.float32)
            losses.append(loss.losses.ring_loss(flags, lenet_fc))

            tf.summary.scalar(flags.op.name, flags)
            tf.losses.add_loss(losses[-1])
    elif loss_id == 'radial_loss':
        with tf.variable_scope('radial_loss') as scope:
            losses.append(loss.losses.radial_loss(labels, lenet_fc))
            tf.losses.add_loss(losses[-1])

    # combine losses into one:
    with tf.variable_scope('joint_loss') as scope:
        if len(losses) == 1:
            joint_loss = losses[0]
        elif isinstance(loss_weights, list) or isinstance(loss_weights, np.ndarray):
            # NOTE sums by nonzero weights
            print(f'loss weights = {loss_weights}\nof type {type(loss_weights)}')

            loss_weights = tf.constant(loss_weights, dtype=tf.float32, name='Const_loss_weights')

            joint_loss = tf.losses.compute_weighted_loss(
                tf.stack(losses),
                loss_weights,
                # scope,
            )
        elif joint_loss_method == 'sum':
            joint_loss = sum(losses)
        elif joint_loss_method == 'arithmetic_mean':
            joint_loss = tf.reduce_mean(tf.stack(losses), name='arithmetic_mean_of_losses')
        elif joint_loss_method == 'geometric_mean':
            joint_loss = tf.reduce_prod(tf.stack(losses), name='geometric_mean_of_losses') ** (1 / len(losses))

    # tf.summary.scalar('joint_loss', joint_loss)
    loss_avg_op = loss_summaries(joint_loss)

    # Create optimizer and training op.
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    gradients = optimizer.compute_gradients(joint_loss)

    nones = [g[1] for g in gradients if g[0] is None]
    for none in nones:
        print(none)

    for i, g in enumerate(gradients):
        # tf.keras.backend.print_tensor(g)
        tf.summary.histogram(f'{g[1].name}-grad', g[0])

    train_op = optimizer.apply_gradients(gradients, global_step)

    for var in tf.trainable_variables():
        if 'conv' in var.name:
            tf.summary.histogram(var.name, var)
        elif 'fully_connected' in var.name:
            tf.summary.histogram(var.name, var)

    # specify metrics
    accuracy, acc_op = tf.metrics.accuracy(
        tf.argmax(labels, 1),
        tf.argmax(lenet_output, 1),
    )
    tf.summary.scalar('accuracy', acc_op)
    summary_op = tf.summary.merge_all()

    with tf.Session(config=sess_config) as sess:
        # Build summary operation
        summary_writer = tf.summary.FileWriter(
            os.path.join(
                output_dir,
                'summaries',
                summary_path,
                str(datetime.now()).replace(':', '-').replace(' ', '_'),
            ),
            sess.graph
        )

        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        print('Begin Global Step Loop.')

        # Train model
        count = 0
        batch_count = math.ceil(len(y_train) / batch_size)
        for epoch in range(epochs):
            print(f'Epoch {epoch+1} / {epochs}')
            print(f'Global step {count}.')
            if count > 0:
                print(f'joint_loss = {train_results["joint_loss"]}')
                print(f'softmax = {train_results["softmax"]}')
                print(f'last_loss = {train_results["last_loss"]}')
                print(f'IS accuracy = {train_results["acc_op"]}')

            # go through all batches
            for j in range(batch_count):
                # print(f'Batch {j+1} / {batch_count}.')
                train_results = sess.run({
                    'train_op': train_op,
                    'joint_loss': joint_loss,
                    'lenet_output': lenet_output,
                    'acc_op': acc_op,
                    'lenet_fc': lenet_fc,
                    'softmax': losses[0],
                    'last_loss': losses[-1],
                    'loss_avg_op': loss_avg_op,
                    'summary_op': summary_op,
                })
                count += 1

                summary_writer.add_summary(train_results['summary_op'], count)
                summary_writer.flush()

        summary_writer.close()

        if output_dir:
            output_dir = os.path.join(
                output_dir,
                str(datetime.now()).replace(':', '-').replace(' ', '_'),
            )
            os.makedirs(output_dir, exist_ok=True)

            train_results.pop('summary_op')

            save_json(
                os.path.join(output_dir, 'train_results.json'),
                train_results,
                summary_params,
            )

            # save model variables:
            tf.saved_model.simple_save(
                sess,
                os.path.join(output_dir, 'trained_model_vars.ckpt'),
                {'features': features, 'labels': labels},
                {'lenet_output': lenet_output},
            )

        # Predict with model
        test_results = sess.run(
            {
                'joint_loss': joint_loss,
                'lenet_output': lenet_output,
                'acc_op': acc_op,
                'lenet_fc': lenet_fc,
                'softmax': losses[0],
                'last_loss': losses[-1],
                'loss_avg_op': loss_avg_op,
            },
            {features: x_test, labels: y_test},
        )

        if output_dir:
            save_json(
                os.path.join(output_dir, 'test_results.json'),
                test_results,
                summary_params,
            )

    return train_results, test_results


def save_json(filepath, results, additional_info=None, deep_copy=True):
    if deep_copy:
        results = deepcopy(results)
    if additional_info:
        results.update(additional_info)

    # np_dir = os.path.join(filepath.rpartition(os.path.sep)[0], 'np_files')

    with open(filepath, 'w') as summary_file:
        for key in results:
            if isinstance(results[key], np.ndarray):
                # save to numpy specifc dir and save as csv.
                results[key] = results[key].tolist()
            elif isinstance(results[key], np.integer):
                results[key] = int(results[key])
            elif isinstance(results[key], np.floating):
                results[key] = float(results[key])

        json.dump(results, summary_file, indent=4, sort_keys=True)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Run proof of concept ')

    parser.add_argument('-d', '--dense_size', default=500, type=int, help='The number of units in dense layer of letnet.')

    parser.add_argument('-b', '--batch_size', default=1000, type=int, help='The number of units in dense layer of letnet.')
    parser.add_argument('-e', '--epochs', default=1, type=int, help='The number of epochs.')

    parser.add_argument('-l', '--loss_id', default='softmax', help='The loss to use', choices=['softmax', 'center_loss', 'tcl', 'tcl_softmax', 'objectosphere'])

    parser.add_argument('-t', '--num_target_labels', default=10, type=int, help='The number of target labels.')

    parser.add_argument('-j', '--joint_loss_method', default='arithmetic_mean', help='The method used to combine the losses together.', choices=['arithmetic_mean', 'sum', 'geometric_mean'])

    parser.add_argument('-m', '--model_id', default='lenet++', help='The model to use', choices=['lenet', 'lenet++'])

    parser.add_argument('-o', '--output_dir', default='./', help='Filepath to the output directory.')
    parser.add_argument('-s', '--summary_path', default='', help='Filepath appened to the output directory for saving the summaries.')
    parser.add_argument('-r', '--random_seed', default=None, type=int, help='The random seed to use for initialization of the model.')
    parser.add_argument('-L', '--loop', default=0, type=int, help='The number of times to run the things.')

    # parser.add_argument('-w', '--loss_weights', default=None, type=json.loads, help='Loss weights as a dict')
    parser.add_argument('-w', '--loss_weights', nargs='+', help='Loss weights in order of losses.')
    parser.add_argument('--margin', default=0, type=float, help='The margin parameters.')

    # Hardware
    parser.add_argument('--cpu', default=1, type=int, help='The number of available CPUs.')
    parser.add_argument('--cpu_cores', default=1, type=int, help='The number of available cores per CPUs.')
    parser.add_argument('--gpu', default=0, type=int, help='The number of available GPUs. Pass negative value if no CUDA.')
    parser.add_argument('--which_gpu', default=None, type=int, help='The number of available GPUs. Pass negative value if no CUDA.')

    parser.add_argument('--shuffle_data', action='store_false', help='Disable shuffling of data.')

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.loss_id, args.joint_loss_method)
    os.makedirs(args.output_dir, exist_ok=True)
    print(args.output_dir)

    if isinstance(args.loss_weights, list):
        print(args.loss_weights)
        args.loss_weights = np.array(args.loss_weights, dtype=np.float32)

    # Set the Hardware
    config = tf.ConfigProto(
        intra_op_parallelism_threads=args.cpu_cores,
        inter_op_parallelism_threads=args.cpu_cores,
        allow_soft_placement=True,
        device_count={
            'CPU': args.cpu,
            'GPU': args.gpu,
        } if args.gpu >= 0 else {'CPU': args.cpu},
    )

    params = vars(args)
    params['sess_config'] = config

    print('shuffle_data = ', params['shuffle_data'])

    if args.which_gpu:
        # NOTE does not work atm.
        with tf.device(f'/gpu:{args.which_gpu}'):
            results = run_lenet(**params)
    else:
        if args.loop > 1:
            for i in range(args.loop):
                results = run_lenet(**params)
        else:
            results = run_lenet(**params)
