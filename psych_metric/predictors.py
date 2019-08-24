"""Simple testing concept."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
import random
from time import perf_counter, process_time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold

from psych_metric.datasets import data_handler


def run_experiment(
        output_dir,
        label_src,
        dataset_id,
        data_config,
        model_config,
        kfold_cv_args):
    # Load and prep dataset
    dataset = data_handler.load_dataset(dataset_id, **data_config)
    images, labels = dataset.load_images()

    # select the label source for this run
    if label_src == 'annotations':
        labels = dataset.df

        # TODO handle proper binariing of annotations labels.
        raise NotImplementedError
    elif label_src == 'majority_vote' or label_src == 'ground_truth':
        labels = labels[label_src]

        # Binarize the label data
        label_bin = LabelBinarizer()
        label_bin.fit(labels)
        y_data = label_bin.transform(labels).astype('float32', copy=False)
    else:
        raise ValueError(
            'expected `label_src` to be "majority_vote", "ground_truth", or '
            + f'"annotations", but recieved {label_src}',
        )

    summary = {
        dataset_id: data_config,
        'model_config': model_config,
        'kfold_cv_args': kfold_cv_args,
        'label_binarizer': label_bin.classes_.tolist()
    }

    kfold_cv(
        model_config,
        images,
        y_data,
        output_dir,
        summary=summary,
        **kfold_cv_args,
    )


def kfold_cv(
        model_config,
        features,
        labels,
        output_dir,
        summary,
        kfolds=5,
        random_seed=None,
        save_pred=True,
        save_model=True,
        stratified=None,
        test_focus_fold=True,
        shuffle=True,
        repeat=None):
    """Generator for kfold cross validation.

    Parameters
    ----------
    model :
        custom class or something needs: fit/train, eval, save
    data :
        The data to be split into folds and used in cross validation.
    k : int, optional
        The number of folds.
    random_seed : int, optional
        The seed used for initializing the random number generators.
    save_pred : bool, optional
        returns the predictions in addition to the rest of the output.
    save_model : bool, optional
        returns the model in addition to the rest of the output.
    stratified : bool | array-like, optional
        By default the data splitting is unstratified (default is None). If
        True, the data is stratified when split to preserve class balance of
        original dataset. If a 1-D array-like object of same length as data,
        then it is treated as the strata of the data for splitting.
    test_focus_fold : bool, optional
        If True (default), the single focus fold will be the current
        iteration's test set while the rest are used for training the model,
        otherwise the focus fold is the only fold used for training and the
        rest are used for testing.
    """
    if not random_seed:
        random_seed = random.randint(0, 2**32 - 1)

        # TODO use same seed for initializing the model everytime or different seeds?
        raise NotImplementedError
        # shuffle param indicates if they want data shuffling or not, ie. seed to be made or not for shuffling only.

    # Create the kfolds directory for this experiment.
    output_dir_kfolds = os.path.join(output_dir, f'{kfolds}_fold_cv')
    os.makedirs(output_dir_kfolds, exist_ok=True)

    # Data index splitting
    if stratified:
        fold_indices = StratifiedKFold(kfolds, shuffle, random_seed).split(features, labels)
    else:
        fold_indices = KFold(kfolds, shuffle, random_seed).split(features)

    for i, (other_folds, focus_fold) in enumerate(fold_indices):
        if random_seed:
            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)

        kfold_summary = summary.copy()
        kfold_summary['kfold_cv_args'].update({
            'random_seed': random_seed,
            'focus_fold': i + 1,
        })

        output_dir_eval_fold = os.path.join(output_dir_kfolds, f'eval_fold_{i+1}')
        os.makedirs(output_dir_eval_fold, exist_ok=True)

        # Set the correct train and test indices
        if test_focus_fold:
            train_idx = other_folds
            test_idx = focus_fold
        else:
            train_idx = focus_fold
            test_idx = other_folds

        logging.info(f'{i}/{kfolds} fold cross validation: Training')

        model, init_times, train_times = prepare_model(
            model_config,
            features[train_idx],
            labels[train_idx],
            output_dir_eval_fold,
        )

        logging.info(f'{i}/{kfolds} fold cross validation: Testing')

        start_process_time = process_time()
        start_perf_time = perf_counter()

        pred = model.predict(
            features[test_idx],
            **model_config['test'] if 'test' in model_config else {}
        )

        test_process = process_time() - start_process_time
        test_perf = perf_counter() - start_perf_time

        if save_pred:
            np.savetxt(
                os.path.join(output_dir_eval_fold, 'pred.csv'),
                pred,
                delimiter=',',
            )

        kfold_summary['runtimes'] = {
            'init': init_times,
            'train': train_times,
            'test': {
                'process': test_process,
                'perf': test_perf,
            },
        }

        # save summary
        save_json(
            os.path.join(output_dir_eval_fold, 'summary.json'),
            kfold_summary,
        )


def prepare_model(model_config, features, labels, output_dir=None):
    if 'filepath' in model_config:
        # TODO could remove from here and put into calling code when loading is possible
        return tf.keras.models.load_model(model_config['filepath']), None, None

    start_process_time = process_time()
    start_perf_time = perf_counter()

    model = load_model(model_config['model_id'], **model_config['init'])

    init_process = process_time() - start_process_time
    init_perf = perf_counter() - start_perf_time

    start_process_time = process_time()
    start_perf_time = perf_counter()

    model.fit(
        features,
        labels,
        **model_config['train'] if 'train' in model_config else {}
    )

    train_process = process_time() - start_process_time
    train_perf = perf_counter() - start_perf_time

    if isinstance(output_dir, str) and os.path.isdir(output_dir):
        model.save(os.path.join(output_dir, f'{model_config["model_id"]}.h5'))

    init_times = {'process': init_process, 'perf': init_perf}
    train_times = {'process': train_process, 'perf': train_perf}

    return model, init_times, train_times


def load_model(model_id, crowd_layer=False, **kwargs):
    if model_id.lower() == 'vgg16':
        model = vgg16_model(crowd_layer=crowd_layer)

        if crowd_layer:
            # TODO model.compile('adam', CrowdLayer...)
            raise NotImplementedError
        else:
            model.compile('adam', 'categorical_crossentropy')
    if model_id.lower() == 'resnext50':
        model = resnext50_model(crowd_layer=crowd_layer)

        if crowd_layer:
            # TODO model.compile('adam', CrowdLayer...)
            raise NotImplementedError
        else:
            model.compile('adam', 'categorical_crossentropy')

    return model


def vgg16_model(input_shape=(256, 256, 3), num_labels=8, crowd_layer=False):
    input_layer = tf.keras.layers.Input(shape=input_shape, dtype='float32')

    # create model and freeze them model.
    vgg16 = tf.keras.applications.vgg16.VGG16(False, input_tensor=input_layer)
    for layer in vgg16.layers:
        layer.trainable = False

    # Add the layers specified in Crowd Layer paper.
    x = tf.keras.layers.Flatten()(vgg16.layers[-1].output)
    x = tf.keras.layers.Dense(128, 'relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_labels, 'softmax')(x)

    if crowd_layer:
        # TODO: Crowd Layer for the VGG16 model.
        raise NotImplementedError

    return tf.keras.models.Model(inputs=input_layer, outputs=x)


def resnext50_model(input_shape=(256, 256, 3), crowd_layer=False):
    input_layer = tf.keras.layers.Input(shape=input_shape, dtype='float32')

    # create model and freeze them model.
    resnext50 = tf.keras.applications.resnext.ResNeXt50(input_tensor=input_layer)

    # TODO need to do a thing to make the model for the dataset.

    if crowd_layer:
        # TODO: Crowd Layer for the VGG16 model.
        raise NotImplementedError

    return tf.keras.models.Model(inputs=input_layer, outputs=resnext50)


def save_json(filepath, results, additional_info=None, deep_copy=True):
    if deep_copy:
        results = deepcopy(results)
    if additional_info:
        results.update(additional_info)

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
    parser = argparse.ArgumentParser(description='Run proof of concept ')

    # Model args.
    parser.add_argument(
        '-m',
        '--model_id',
        default='vgg16',
        help='The model to use',
        choices=['vgg16', 'resnext50'],
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        default=32,
        type=int,
        help='The number of units in dense layer of letnet.'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        default=1,
        type=int,
        help='The number of epochs.',
    )
    parser.add_argument(
        '-r',
        '--random_seed',
        default=None,
        type=int,
        help='The random seed to use for initialization of the model.',
    )

    # Data args
    parser.add_argument(
        '-d',
        '--dataset_id',
        default='LabelMe',
        help='The dataset to use',
        choices=['LabelMe', 'FacialBeauty', 'All_Ratings'],
    )
    parser.add_argument(
        'dataset_filepath',
        help='The filepath to the data directory',
    )
    parser.add_argument(
        '-l',
        '--label_src',
        default='majority_vote',
        help='The source of labels to use for training.',
        choices=['majority_vote', 'ground_truth', 'annotations'],
    )

    # Output args
    parser.add_argument(
        '-o',
        '--output_dir',
        default='./',
        help='Filepath to the output directory.',
    )
    parser.add_argument(
        '-s',
        '--summary_path',
        default='',
        help='Filepath appened to the output directory for saving the summaries.',
    )

    # Hardware
    parser.add_argument(
        '--cpu',
        default=1,
        type=int,
        help='The number of available CPUs.',
    )
    parser.add_argument(
        '--cpu_cores',
        default=1,
        type=int,
        help='The number of available cores per CPUs.',
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )
    parser.add_argument(
        '--which_gpu',
        default=None,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )

    # K Fold CV args
    parser.add_argument(
        '-k',
        '--kfolds',
        default=5,
        type=int,
        help='The number of available CPUs.',
    )
    parser.add_argument(
        '--shuffle_data',
        action='store_false',
        help='Disable shuffling of data.',
    )
    parser.add_argument(
        '--crowd_layer',
        action='store_true',
        help='Use crowd layer in ANNs.',
    )
    parser.add_argument(
        '--no_save_pred',
        action='store_false',
        help='Predictions will not be saved.',
    )
    parser.add_argument(
        '--no_save_model',
        action='store_false',
        help='Model will not be saved.',
    )
    parser.add_argument(
        '--stratified',
        action='store_true',
        help='Stratified K fold cross validaiton will be used.',
    )
    parser.add_argument(
        '--train_focus_fold',
        action='store_true',
        help='The focus fold in K fold cross validaiton will be used for '
        + 'training and the rest will be used for testing..',
    )

    args = parser.parse_args()

    # create identifying directory path for saving results.
    args.output_dir = os.path.join(
        args.output_dir,
        args.dataset_id,
        args.model_id,
        str(args.random_seed),
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    )

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

    # params = vars(args)
    # params['sess_config'] = config
    tf.keras.backend.set_session(tf.Session(config=config))

    # package the arguements:
    data_config = {'dataset_filepath': args.dataset_filepath}

    model_config = {
        'model_id': args.model_id,
        'init': {'crowd_layer': args.crowd_layer},
        'train': {'epochs': args.epochs, 'batch_size': args.batch_size},
    }

    kfold_cv_args = {
        'kfolds': args.kfolds,
        'random_seed': args.random_seed,
        'save_pred': args.no_save_pred,
        'save_model': args.no_save_model,
        'stratified': args.stratified,
        'test_focus_fold': not args.train_focus_fold,
        'shuffle': args.shuffle_data,
        # 'repeat': None,
    }

    if args.which_gpu:
        # TODO does not work atm...
        raise NotImplementedError('Selecting specific GPU not implemented.')
    else:
        run_experiment(
            args.output_dir,
            args.label_src,
            args.dataset_id,
            data_config,
            model_config,
            kfold_cv_args,
        )
