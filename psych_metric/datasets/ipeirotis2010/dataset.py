"""Dataset class handler for Ipeirotis2010 data."""
import os
import csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from psych_metric.datasets.base_dataset import BaseDataset

ROOT = os.environ['ROOT']
HERE = os.path.join(ROOT, 'psych_metric/datasets/ipeirotis2010/')

class Ipeirotis2010(BaseDataset):
    """class that loads and serves data from Ipeirotis 2010

    Attributes
    ----------
    dataset : str
        Name of specific dataset
    annotations : pandas.DataFrame
        Data Frame containing annotations
    label_set : set
        Set containing the complete original labels
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    sparse_matrix : bool
        Dataframe uses datafile structure if True, uses sparse matrix format if
        False. Default value is False
    """

    def __init__(self, dataset='AdultContent', encode_columns=None, sparse_matrix=False):
        """initialize class by loading the data

        Parameters
        ----------
        dataset : str
            the name of one of the subdatasets corresponding to file name
        encode_columns : list, optional
            Encodes columns provided as list of str; dataframe uses raw values
            by default.
        sparse_matrix : bool, optional
            Convert the data into a dataframe with the sparse matrix structure
        """
        dsets = [
            'AdultContent', 'AdultContent2', 'AdultContent3',
            'BarzanMozafari', 'CopyrightInfringement',
            'HITspam-UsingCrowdflower', 'HITspam-UsingMTurk',
            'JeroenVuurens'
        ]
        self._check_dataset(dataset, dsets)
        if dataset == 'AdultContent3':
            self.dataset = 'AdultContent3-HCOMP2010'
        else:
            self.dataset = dataset

        if not isinstance(sparse_matrix, bool):
            raise TypeError('sparse_matrix parameter must be a boolean.')
        self.sparse_matrix = sparse_matrix

        # Read in and save data
        if dataset == 'JeroenVuurens':
            annotation_file = os.path.join(HERE, 'ipeirotis2010_data', self.dataset, 'votes')
            self.annotations = pd.read_csv(annotation_file, header=None, names=['time', 'question_id', 'worker_id', 'label'], delimiter='\t')
        else:
            annotation_file = os.path.join(HERE, 'ipeirotis2010_data', self.dataset, 'labels.txt')
            self.annotations = pd.read_csv(annotation_file, header=None, names=['worker_id', 'url', 'label'], delimiter='\t')

        # Read in and save the expected labels, or infer the labels from data
        if 'HITspam' in dataset:
            self.label_set = frozenset({Yes, No})
        elif dataset == 'JeroenVuurens':
            self.label_set = frozenset({0, 1})
        else:
            labels_file = os.path.join(HERE, 'ipeirotis2010_data', self.dataset, 'categories.txt')
            self.label_set = self._load_labels(labels_file)

        # TODO automate the encoding of columns if encode_columns is True:
        # ie. hardcode columns for each dataset to be encoded if True.
        # Encode the labels and data if desired
        if 'AdultContent' in dataset or dataset == 'CopyrightInfringement' or 'HITspam' in dataset:
            if encode_columns == True:
                encode_columns = {'worker_id', 'url', 'label'}
            column_labels = {'label': self.label_set}
        elif dataset == 'BarzanMozafari':
            if encode_columns == True:
                encode_columns = {'worker_id'}
            column_labels = None
        elif dataset == 'JeroenVuurens':
            if encode_columns == True:
                encode_columns = {'question_id'}
            column_labels = None
        else:
            column_labels = None

        self.label_encoder =  None if encode_columns is None else self.encode_labels(encode_columns, column_labels)

        # Restructure dataframe into a sparse matrix
        if sparse_matrix:
            self.annotations = self.convert_to_sparse_matrix(annotations)

    def _load_labels(self, csv_path=None):
        """ Load labels from the published dataset categories file, or infer
        from the current dataframe.

        Parameters
        ----------
        csv_path : str, optinal
            path of categories file to load labels from; infer from data if None

        Returns
        -------
        frozenset
            frozenset of the categories as the label set.
        """
        if csv_path is None:
            # infer labels from data, will exclude any missing labels
            return frozenset(self.annotations['label'].unique())

        # read in the labels provided
        label_set = set()
        with open(csv_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter='\t')
            for row in csv_reader:
                label_set.add(row[0])
        return frozenset(label_set)


    def convert_to_sparse_matrix(self, annotations):
        """Convert provided dataframe into a sparse matrix equivalent.

        Converts the given dataframe of expected format equivalent to this
        dataset's datafile structure into a sparse matrix dataframe where the
        row corresponds to each sample and the columns are structured as
        features, annotations of worker_id, where missing annotations are NA.

        Parameters
        ----------
        annotations : pandas.DataFrame
            Dataframe to be converted into a sparse matrix format.

        Returns
        -------
        pandas.DataFrame
            Data Frame of annotations in a sparse matrix format.

        """
        #TODO decide if this is desireable, then implement if it is desireable.
        raise NotImplementedError

    def __len__(self):
        """ get size of dataset

        Returns
        -------
        int
            number of annotations(rows) in dataset. If sparse matrix, number of
            samples
        """
        return len(self.annotations)

    def __getitem__(self, i):
        """ get specific row from dataset

        Returns
        -------
        dict:
            {header: value, header: value, ...}
        """
        row = self.annotations.iloc[i]
        return dict(row)
