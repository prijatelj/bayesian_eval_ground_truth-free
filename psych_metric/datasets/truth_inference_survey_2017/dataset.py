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
        # This is desirable from a standardization perspective of making the
        # data fit the format of model functions such as those in scikit learn.
        raise NotImplementedError

    def truth_inference_survey_format(self):
        """Necessary for converting the dataframe list format into what the
        Truth Inference Survey's code expects.

        Returns
        -------
            Tuple of a dictionary of sample identifiers as keys and values as list of
            annotator id and their annotation, and a dictionary of annotator
            identifeirs as keys and values as list of samples.
        """
        #TODO create a generalized version of this that would apply to any data.
        samples_to_annotations = dict()
        annotators_to_samples = dict()

        for index, row in self.df.iterrows():
            # add to samples_to_annotations
            if row['sample_id'] not in samples_to_annotations:
                samples_to_annotations[row['sample_id']] = [row['worker_id'], row['label']]
            else:
                samples_to_annotations[row['sample_id']].append([row['worker_id'], row['label']])

            # add to annotators_to_samples
            if row['worker_id'] not in annotators_to_samples:
                annotators_to_samples[row['worker_id']] = [row['sample_id'], row['label']]
            else:
                annotators_to_samples[row['worker_id']].append([row['sample_id'], row['label']])

        return samples_to_annotations, annotators_to_samples

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
