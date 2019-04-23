"""Dataset class handler for truth inference survey 2017 data."""
import os

import numpy as np
import pandas as pd

from psych_metric.datasets.base_dataset import BaseDataset

ROOT = os.environ['ROOT']
HERE = os.path.join(ROOT, 'psych_metric/datasets/truth_inference_survey_2017/truth_inference_survey_2017_data/')

class TruthSurvey2017(BaseDataset):
    """class that loads and serves data from truth inference survey 2017

    Attributes
    ----------
    dataset : str
        Name of specific (sub) dataset contained within this class.
    df : pandas.DataFrame
        contains the data of the dataset in a standardized format (typically
        an annotation list where each row is an individual's annotation of one
        sample. Must contain the columns: 'worker_id' and 'label'.
        'ground_truth' is also a common column name when ground truth is
        included with the original dataset. 'sample_id' will exist when no
        features are provided, or some features need loaded.
    label_set : set
        Set containing the complete original labels.
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    sparse_matrix : bool
        Dataframe uses datafile structure if True, uses sparse matrix format if
        False. Default value is False
    """

    def __init__(self, dataset='d_Duck Identification', encode_columns=None):
        """initialize class by loading the data

        Parameters
        ----------
        dataset : str
            the name of one of the subdatasets corresponding to file name
        encode_columns : list, optional
            Encodes columns provided as list of str; dataframe uses raw values
            by default.
        """
        self.load_dataset(dataset, encode_columns)

    def load_dataset(dataset='d_Duck Identification', encode_columns=None):
        dsets = [
            'd_Duck Identification', 'd_jn-product', 'd_sentiment',
            's4_Dog data', 's4_Face Sentiment Identification', 's4_Relevance',
            's5_AdultContent', 'f201_Emotion_FULL'
        ]
        self._check_dataset(dataset, dsets)
        self.dataset = dataset

        # Read in and save data
        annotation_file = os.path.join(HERE, self.dataset, 'answer.csv')
        self.df = pd.read_csv(annotation_file)
        # change to standardized column names. #sample_id is a feature.
        self.df.columns = ['sample_id', 'worker_id', 'label']

        labels_file = os.path.join(HERE, self.dataset, 'truth.csv')
        ground_truth = pd.read_csv(labels_file)
        # change to standardized column names.
        ground_truth.columns = ['sample_id', 'label']

        # Save labels set
        self.labels_set = set(ground_truth['label'].unique()) if dataset != 'f201_Emotion_FULL' else None

        # Add ground_truth to the main dataframe as its own column
        self.add_ground_truth_to_samples(ground_truth)

        if encode_columns == True:
            # The default columns to encode for each data subset.
            if dataset in {'d_jn-product', 's4_Relevance', 's5_AdultContent'}:
                encode_columns = {'sample_id', 'worker_id'}
            elif dataset in {'d_sentiment', 's4_Face Sentiment Identification', 'f201_Emotion_FULL'}:
                encode_columns = {'worker_id'}

        # Load and save the features if any.
        self.features = None

        # Encode the labels and data if desired.
        self.label_encoder = None if encode_columns is None else self.encode_labels(encode_columns)

    def get_ground_truth(self, sample_id, decode=False)
        if ground_truth is None:
            return None

        # ground_truth is a dict of raw (decoded) sample_ids
        if decode:
            sample_id = self.label_encoder['sample_id'].inverse_transform(sample_id)

        return self.ground_truth[sample_id]

    def add_ground_truth_to_samples(self, ground_truth, inplace=True, is_dict=False):
        """ Add the ground truth labels to every sample (row) of the main
        dataframe; in place by default.

        Parameters
        ----------
        ground_truth : pandas.DataFrame
            Dataframe of the ground truth,
        inpalce : bool, optinal
            Update dataframe in place if True, otherwise return the updated
            dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with a ground truth labels column returned if inplace is
            False, otherwise returns None.
        """
        if not is_dict:
            # converts to dict first, may or may not be efficent.
            ground_truth_dict = {}
            for i in range(len(self.ground_truth)):
                ground_truth_dict[self.ground_truth.iloc[i,0]] = self.ground_truth.iloc[i,1]
            ground_truth = ground_truth_dict

        ground_truth_col = self.df['sample_id'].apply(lambda x: self.ground_truth[x])

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
        features, df of worker_id, where missing annotations are NA.

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
