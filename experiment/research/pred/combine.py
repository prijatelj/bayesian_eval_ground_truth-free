import argparse
import json

import numpy as np

from experiment import io

def combine(
    in_file1,
    in_file2,
    out_file,
    shuffle=False,
    keys=['conditionals', 'givens'],
):
    """Converts format from `load_fold_save_pred` to given and conditional of
    a single data split (train XOR test)
    """
    with open(in_file1, 'r') as f:
        inf1 = json.load(f)

    with open(in_file2, 'r') as f:
        inf2 = json.load(f)

    for key in keys:
        inf1[key] = np.vstack((np.array(inf1[key]), np.array(inf2[key])))

    if shuffle:
        raise NotImplementedError('Shuffling is not implemented yet.')

    io.save_json(out_file, inf1)


def parse_args():
    parser = argparse.ArgumentParser(description='Combines given JSONs')

    parser.add_argument(
        'in_file1',
        help='The JSON file in format of `save_pred.load_fold_save_pred()`.',
    )

    parser.add_argument(
        'in_file2',
        help='The JSON file in format of `save_pred.load_fold_save_pred()`.',
    )

    parser.add_argument(
        'out_file',
        help='The output file path of converted json.',
    )

    parser.add_argument(
        '--keys',
        default=['conditionals', 'givens'],
        nargs='+',
        type=str,
        help='Keys in json to split in half.',
    )

    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffles the data after combining.',
    )

    return parser.parse_args()


if __name__ == '__main__':
    combine(**vars(parse_args()))
