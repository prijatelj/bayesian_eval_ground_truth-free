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
        Name of specific dataset
    annotations : pandas.DataFrame
        Data frame containing annotations either as an annotation list or
        matrix. If an annotation list, the columns are sample_id, worker_id, and
        label with rows representing a single annotation from one worker.
    features : pandas.DataFrame
        The dataframe containing the sample ids and their features.
    ground_truth : pandas.DataFrame
        Data Frame containing the ground truth label data
    label_set : set
        Set containing the complete original labels
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
        sparse_matrix : bool, optional
            Convert the data into a dataframe with the sparse matrix structure
        samples_with_ground_truth : bool, optional
           Add the ground truth labels to the data samples.
        """
        self.load_dataset(dataset, encode_columns, sparse_matrix, samples_with_ground_truth)

    def load_dataset(dataset='d_Duck Identification', encode_columns=None):
        dsets = [
            'd_Duck Identification', 'd_jn-product', 'd_sentiment',
            's4_Dog data', 's4_Face Sentiment Identification', 's4_Relevance',
            's5_AdultContent', 'f201_Emotion_FULL'
        ]
        self._check_dataset(dataset, dsets)
        self.dataset = dataset

        #if not isinstance(sparse_matrix, bool):
        #    raise TypeError('sparse_matrix parameter must be a boolean.')
        #self.sparse_matrix = sparse_matrix

        # Read in and save data
        annotation_file = os.path.join(HERE, self.dataset, 'answer.csv')
        self.annotations = pd.read_csv(annotation_file)
        # change to standardized column names.
        self.annotations.columns = ['sample_id', 'worker_id', 'label']

        labels_file = os.path.join(HERE, self.dataset, 'truth.csv')
        self.ground_truth = pd.read_csv(labels_file)
        # change to standardized column names.
        self.ground_truth.columns = ['sample_id', 'label']

        # Save labels set
        self.labels_set = set(self.ground_truth['label'].unique()) if dataset != 'f201_Emotion_FULL' else self.labels_set = None

        # Convert ground_truth from dataframe into dict
        ground_truth_dict = {}
        for i in range(len(self.ground_truth)):
            ground_truth_dict[self.ground_truth.iloc[i,0]] = self.ground_truth.iloc[i,1]
        self.ground_truth = ground_truth_dict

        if encode_columns == True:
            if dataset in {'d_jn-product', 's4_Relevance', 's5_AdultContent'}:
                encode_columns = {'sample_id', 'worker_id'}
            elif dataset in {'d_sentiment', 's4_Face Sentiment Identification', 'f201_Emotion_FULL'}:
                encode_columns = {'worker_id'}

        # Load and save the features if any.
        # NOTE none of these datasets have the features except for Adult, which is just the URL
        self.features = None

        # Encode the labels and data if desired NOTE only encodes annotation.
        self.label_encoder = None if encode_columns is None else self.encode_labels(encode_columns)

        # Restructure dataframe into a sparse matrix
        #if sparse_matrix:
        #    self.annotations = self.convert_to_sparse_matrix(annotations)

    def get_ground_truth(self, sample_id, decode=False)
        if ground_truth is None:
            return None

        # ground_truth is a dict of raw (decoded) sample_ids
        if decode:
            sample_id = self.label_encoder['sample_id'].inverse_transform(sample_id)

        return self.ground_truth[sample_id]

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

        ground_truth = self.annotations['sample_id'].apply(lambda x: self.ground_truth[x])

        # NOTE is_annotation_list does not accept ground_truth in column of dataframe.
        if inplace:
            self.annotations['ground_truth'] = ground_truth
        else:
            annotations_copy = self.annotations.copy()
            annotations_copy['ground_truth'] = ground_truth
            return annotations_copy

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
