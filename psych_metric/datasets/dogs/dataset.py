"""Dataset class handler for Dog classification."""
import os
#import numpy as np
import pandas as pd

from psych_metric.datasets.base_dataset import BaseDataset

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric/datasets/dogs/')
except KeyError:
    HERE = None

class Dogs(BaseDataset):
    """class that loads and serves data from truth inference survey 2017

    Attributes
    ----------
    task_type : str
        The type of learning task the dataset is intended for. This can be one
        of the following: 'regression', 'binary_classification', 'classification'
    images:
    labels:
    label_set : set
        Set containing the complete original labels.
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    """
    def __init__(self, dataset=None, dataset_filepath=None, encode_columns=None, ground_truth=False, tf_load=False, preprocess=False):
        """initialize class by loading the data

        Parameters
        ----------
        dataset : str
            the name of one of the subdatasets corresponding to file name
        encode_columns : list, optional
            Encodes columns provided as list of str; dataframe uses raw values
            by default.
        """
        self._check_dataset(dataset, TruthSurvey2017.datasets)
        self.dataset = dataset

        if dataset_filepath is None:
            if HERE is None or 'images' not in HERE:
                raise ValueError(f"""A path to the dataset file was not provided
                either by the `dataset_filepath` parameter or by the ROOT
                environment variable. Global variable HERE is `{HERE}`. It is
                recommended to use the `dataset_filepath` parameter to provide
                the filepath.""")
            dataset_filepath = HERE
        self.data_dir = dataset_filepath

        self.task_type = 'classification'

        if tf_load:

        else:

    def load_file(dataset, dataset_filepath, ground_truth=False):
        # Read in and save data
        annotation_file = os.path.join(dataset_filepath, dataset, 'answer.csv')
        self.df = pd.read_csv(annotation_file)
        self.df.columns = ['sample_id', 'worker_id', 'worker_label']

        labels_file = os.path.join(dataset_filepath, self.dataset, 'truth.csv')
        ground_truth_df = pd.read_csv(labels_file)
        ground_truth_df.columns = ['sample_id', 'worker_label']

        self.label_set = set(ground_truth_df['worker_label'].unique())

        # Add ground_truth to the main dataframe as its own column
        if ground_truth:
            self.add_ground_truth(ground_truth_df, inplace=True)

        # TODO load images


        # Encode the labels and data if desired.
        self.label_encoder = None if encode_columns is None else self.encode_labels(encode_columns)

    def preprocess(images, bounding_box_file, output=None):
        """Preprocess by the bounding box files."""


        if output:


    def load_tfrecords():

    def __len__(self):
        """ get size of dataset

        Returns
        -------
        int
            number of annotations(rows) in dataset. If sparse matrix, number of
            samples
        """
        return len(self.df)

    def __getitem__(self, i):
        """ get specific row from dataset

        Returns
        -------
        dict:
            {header: value, header: value, ...}
        """
        row = self.df.iloc[i]
        return dict(row)
