"""Dataset class handler for TREC Relevancy 2010 data."""
import os

import numpy as np
import pandas as pd

from psych_metric.datasets.base_dataset import BaseDataset

ROOT = os.environ['ROOT']
HERE = os.path.join(ROOT, 'psych_metric/datasets/trec_relevancy2010/trec_relevancy2010_data/')

class TRECRelevancy2010(BaseDataset):
    """class that loads and serves data from TREC Relevancy 2010

    Attributes
    ----------
    dataset : str
        Name of specific dataset
    df : pandas.DataFrame
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

    def __init__(self, encode_columns=None, sparse_matrix=False):
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
        self.dataset = 'trec-rf10-data'

        if not isinstance(sparse_matrix, bool):
            raise TypeError('sparse_matrix parameter must be a boolean.')
        self.sparse_matrix = sparse_matrix

        # Read in and save data
        annotation_file = os.path.join(HERE, self.dataset + '.txt')
        self.df = pd.read_csv(annotation_file, delimiter='\t')

        # Save the expected labels, or infer the labels from data
        self.label_set = frozenset({-2, -1, 0, 1, 2})
        if encode_columns == True:
            encode_columns = {'docID'}

        # NOTE the 'w' prefix to all workerIDs is removed to use them as integers
        self.df['workerID'] = self.df['workerID'].apply(lambda x: int(x[1:]))

        # TODO automate the encoding of columns if encode_columns is True:
        # ie. hardcode columns for each dataset to be encoded if True.
        # Encode the labels and data if desired
        self.label_encoder =  None if encode_columns is None else self.encode_labels(encode_columns)

        # Restructure dataframe into a sparse matrix
        if sparse_matrix:
            self.df = self.convert_to_sparse_matrix(self.df)

    def convert_to_sparse_matrix(self, df):
        """Convert provided dataframe into a sparse matrix equivalent.

        Converts the given dataframe of expected format equivalent to this
        dataset's datafile structure into a sparse matrix dataframe where the
        row corresponds to each sample and the columns are structured as
        features, annotations of worker_id, where missing annotations are NA.

        Parameters
        ----------
        df : pandas.DataFrame
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
