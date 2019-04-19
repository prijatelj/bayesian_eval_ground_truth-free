"""Dataset class handler for truth inference survey 2010 data."""
import os

import numpy as np
import pandas as pd

from psych_metric.datasets.base_dataset import BaseDataset

ROOT = os.environ['ROOT']
HERE = os.path.join(ROOT, 'psych_metric/datasets/truth_inference_survey_2010/truth_inference_survey_2010_data/')

class TruthSurvey2010(BaseDataset):
    """class that loads and serves data from truth inference survey 2010

    Attributes
    ----------
    dataset : str
        Name of specific dataset
    df : pandas.DataFrame
        Data Frame containing annotations
    label_set : set
        Set containing the complete original labels
    label_df : pandas.DataFrame
        Data Frame containing the ground truth label data
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    sparse_matrix : bool
        Dataframe uses datafile structure if True, uses sparse matrix format if
        False. Default value is False
    """

    def __init__(self, dataset='d_Duck Identification', encode_columns=None, sparse_matrix=False, samples_with_ground_truth=False):
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
        samples_with_ground_truth : bool, optional
           Add the ground truth labels to the data samples.
        """
        dsets = [
            'd_Duck Identification', 'd_jn-product', 'd_sentiment',
            's4_Dog data', 's4_Face Sentiment Identification', 's4_Relevance',
            's5_AdultContent', 'f201_Emotion_FULL'
        ]
        self._check_dataset(dataset, dsets)
        self.dataset = dataset

        if not isinstance(sparse_matrix, bool):
            raise TypeError('sparse_matrix parameter must be a boolean.')
        self.sparse_matrix = sparse_matrix

        # Read in and save data
        annotation_file = os.path.join(HERE, self.dataset, 'answer.csv')
        self.df = pd.read_csv(annotation_file)


        # Save labels set
        if dataset != 'f201_Emotion_FULL'
            labels_file = os.path.join(HERE, self.dataset, 'truth.csv')
            self.labels_df = pd.read_csv(labels_file)
            self.labels_set = set(self.labels_df['truth'].unique())
        else:
            self.labels_set = None

        if encode_columns == True:
            if dataset in {'d_jn-product', 's4_Relevance', 's5_AdultContent'}:
                encode_columns = {'question', 'worker'}
            elif dataset in {'d_sentiment', 's4_Face Sentiment Identification', 'f201_Emotion_FULL'}:
                encode_columns = {'worker'}

        if samples_with_ground_truth:
            self.add_ground_truth_to_samples()

        # TODO automate the encoding of columns if encode_columns is True:
        # ie. hardcode columns for each dataset to be encoded if True.
        # Encode the labels and data if desired
        self.label_encoder = None if encode_columns is None else self.encode_labels(encode_columns)

        # Restructure dataframe into a sparse matrix
        if sparse_matrix:
            self.df = self.convert_to_sparse_matrix(df)

    def add_ground_truth_to_samples(self, inplace=True):
        """ Add the labels to every sample, in place by default.

        Parameters
        ----------
        inpalce : bool, optinal
            Update dataframe in place if True, otherwise return the updated
            dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with a ground truth labels column returned if inplace is
            False.
        """
        ground_truth_dict = self.labels_df.to_dict()

        ground_truth = self.df['answer'].apply(lambda x: ground_truth_dict[x])

        if inplace:
            self.df['ground_truth'] = ground_truth
        else:
            df_copy = self.df.copy()
            df_copy['ground_truth'] = ground_truth
            return df_copy

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
