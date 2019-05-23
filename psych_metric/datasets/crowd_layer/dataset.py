"""Dataset class handler for crowd layer 2018 data."""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from psych_metric.datasets.base_dataset import BaseDataset

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric/datasets/crowd_layer/')
except KeyError:
    # If ROOT environment variable does not exist, set HERE to None
    HERE = None

class CrowdLayer(BaseDataset):
    """class that loads and serves data from crowd layer 2018

    Attributes
    ----------
    dataset : str
        Name of the specific dataset with this data.
    task_type : str
        The type of learning task the dataset is intended for. This can be one
        of the following: 'regression', 'binary_classification', 'classification'
    data_dir : str
        Path to directory of the specific dataset with this data.
    label_set : set
        Set containing the complete original labels
    annotations : pandas.DataFrame
        Data Frame containing the annotators' annotations in a sparse matrix
        with samples as rows and columns as the individual annotators.
    labels: pandas.DataFrame
        Data Frame containing the label data from label.txt
    data: pandas.DataFrame
        Data Frame containing the contents of the data.txt
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    sparse_matrix : bool
        Dataframe uses datafile structure if True, uses sparse matrix format if
        False. Default value is False
    """
    datasets = frozenset({'LabelMe', 'MovieReviews', 'ner-mturk'})

    def __init__(self, dataset='MovieReviews', dataset_filepath=None, encode_columns=None):
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
        self._check_dataset(dataset, CrowdLayer.datasets)
        if dataset_filepath is None:
            if HERE is None or 'crowd_layer' not in HERE:
                raise ValueError('A path to the dataset file was not provided either by the `dataset_filepath` parameter or by the ROOT environment variable. Global variable HERE is `%s`. It is recommended to use the `dataset_filepath` parameter to provide the filepath.' % HERE)
            dataset_filepath = HERE

        # save the data directory
        self.data_dir = os.path.join(dataset_filepath, 'crowd_layer_data')


        # All are matrices are sparse matrices. All rows will always be samples
        self.sparse_matrix = True

        # Read in and save data
        # Save labels set
        if dataset == 'LabelMe':
            self.load_LabelMe()
        elif dataset == 'MovieReviews':
            self.load_MovieReviews()
        elif dataset == 'ner-mturk':
            self.load_ner_mturk()

        # NOTE already encoded, but will need to make an inverse encoder though.
        # TODO automate the encoding of columns if encode_columns is True:
        # ie. hardcode columns for each dataset to be encoded if True.
        # Encode the labels and data if desired
        if encode_columns is None or encode_columns is False or dataset == 'MovieReviews':
            self.label_encoder = None
        else:
            #self.encode_labels(encode_columns)
            raise NotImplementedError('Encoding of the multiclassification classes not yet implemented. It requires possibly redoing the `encode_labels` to take and return a dataframe or overriding it.')

    def _check_datasplit(self, datasplit, dsets={'train', 'valid', 'test'}):
        if datasplit not in dsets:
            raise ValueError(str(datasplit) + ' is not an acceptable split on the dataset `' + self.dataset + '`. Use only the following data splits: ' + str(dsets))

    def load_LabelMe(self, datasplit='train'):
        """Loads the designated split of data from LabelMe into this instance's
            attributes.

        Params
        ------
        datasplit : str
            string indicating which split of the data to load into this instance
            of the class object.
        """
        self.dataset = 'LabelMe'
        # Set the dataset task type
        self.task_type = 'classification'

        self.label_set = frozenset({'highway', 'insidecity', 'tallbuilding', 'street', 'forest', 'coast', 'mountain', 'opencountry'})

        # Ensure valid datasplit value
        self._check_datasplit(datasplit)

        # Load annotations if they exist
        if datasplit != 'train':
            # Warn about lack of annotations
            raise Warning('The datasplit `' + datasplit + '` of the dataset `' + self.dataset + '` is not annotated. Only the `train` datasplit is annotated.')
            self.df = None
        else:
            annotation_file = os.path.join(self.data_dir, self.dataset, 'answers.txt')
            self.df = pd.read_csv(annotation_file, sep=' ', header=None).to_sparse(fill_value=-1)

        # Load/create the Labels dataframe
        labels_file = os.path.join(self.data_dir, self.dataset, 'labels_' + datasplit + '.txt')
        self.labels = pd.read_csv(labels_file, sep=' ', names=['label'])

        label_names_file = os.path.join(self.data_dir, self.dataset, 'labels_' + datasplit + '_names.txt')
        self.labels['label_name'] = pd.read_csv(label_names_file, sep=' ', names=['label_name'])

        filenames_file = os.path.join(self.data_dir, self.dataset, 'filenames_' + datasplit + '.txt')
        self.labels['filename'] = pd.read_csv(filenames_file, sep=' ', names=['filename'])

        # Load the data dataframe/dict/numpy thing....
        # NOTE I have no idea what this is. It's a ragged array, 1000 samples,
        # 200 max columns, format is #:# for each element, no idea what it means.
        data_file = os.path.join(self.data_dir, self.dataset, 'data_' + datasplit + '.txt')
        self.data = pd.read_csv(data_file)

    def load_MovieReviews(self, datasplit='train'):
        """Loads the designated split of data from MovieReviews into this
            instance's attributes.
        """
        self.dataset = 'MovieReviews'

        # Set the dataset task type
        self.task_type = 'regression'
        self.label_set = None

        # Ensure valid datasplit value
        self._check_datasplit(datasplit, {'all', 'train', 'test'})

        # Load annotations if they exist
        if datasplit != 'train':
            # Warn about lack of annotations
            raise Warning('The datasplit `' + datasplit + '` of the dataset `' + self.dataset + '` is not annotated. Only the `train` datasplit is annotated.')
            self.df = None
        else:
            annotation_file = os.path.join(self.data_dir, self.dataset, 'answers.txt')
            self.df = pd.read_csv(annotation_file, sep=' ', header=None).to_sparse(fill_value=-1)

        # ratings are assumed to be the labels file.
        labels_file = os.path.join(self.data_dir, self.dataset, 'ratings_' + datasplit + '.txt')
        self.labels = pd.read_csv(labels_file, sep=' ', names=['label'])

        ids_file = os.path.join(self.data_dir, self.dataset, 'ids_' + datasplit + '.txt')
        self.labels['id'] = pd.read_csv(ids_file, sep=' ', names=['id'])

        # text is the data files.
        data_file = os.path.join(self.data_dir, self.dataset, 'texts_' + datasplit + '.txt')
        self.data = pd.read_csv(data_file, sep=' ', names=['text'])

    def load_ner_mturk(self, datasplit='train'):
        """Loads the designated split of data from MovieReviews into this
            instance's attributes.
        """
        self.dataset = 'ner-mturk'
        # Set the dataset task type
        self.task_type = 'classification'

        self.label_set = frozenset({'O', 'B-LOC', 'B-ORG', 'B-MISC', 'B-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'I-PER'})

        # Ensure valid datasplit value
        self._check_datasplit(datasplit, {'train', 'test', 'ground_truth'})
        # NOTE ground_truth is all annotated labels.

        # Load annotations if they exist
        if datasplit != 'ground_truth':
            # Warn about lack of annotations
            raise ValueError('The ' + datasplit + ' datasplit is not implemented. The trainset and testset data are unclear in how they relate to the annotation data. They probably do not, given difference in size and their sum not equalling that of the ground truth labels. Please only use `ground_truth_seq.csv`.')
            #raise Warning('The datasplit `' + datasplit + '` of the dataset `' + self.dataset + '` is not annotated. Only the `train` datasplit is annotated.')
            self.df = None
        else:
            annotation_file = os.path.join(self.data_dir, self.dataset, 'answers_seq.csv')
            self.df = pd.read_csv(annotation_file, sep=' ', na_values='NA', dtype=str).to_sparse(fill_value=np.nan)
            self.df['sequence'] = self.df['sequence'].astype(int)

        # Load/create the Labels dataframe
        labels_file = datasplit if datasplit == 'ground_truth' else datasplit + 'set'
        labels_file = os.path.join(self.data_dir, self.dataset, labels_file + '_seq.csv')

        # the format of this label one is different due to sequencial samples.
        self.labels = pd.read_csv(labels_file, sep=' ', na_values='NA', dtype=str)

        # TODO encode the labels!

        # There is no general data dataframe for this dataset
        self.data = None

    def ner_mturk_convert_to_annotation_list(self, missing_value=None, inplace=False):
        """Convert provided sparse dataframe into a annotation list equivalent.

        Converts the given dataframe of sparse matrix format into a dataframe of
        equivalent data, but in annotation list format where the rows are the
        different instance of annotations by individual annotators and the
        columns are 'sample_id', 'worker_id', and 'label'.

        Parameters
        ----------
        missing_value :
            The missing_value of the sparse matrix that represents when an
            annotator did not annotate a sample. By default this uses the
            pandas.SparseDataFrame.default_fill_value.
        inplace : bool
            Will update the pandas.DataFrame inplace if True, otherwise returns
            the resulting dataframe.

        Returns
        -------
        pandas.DataFrame
            Data Frame of annotations in an annotation list format.

        """
        if missing_value is None:
            missing_value = self.df.default_fill_value

        num_workers = len(self.df.columns) - 2 # do not count sequence or token
        list_df = pd.DataFrame(np.empty((len(self.df.index)*num_workers, 4)), columns=['sample_id', 'token', 'worker_id', 'worker_label'])

        # Place the correct value in the resulting dataframe list
        for sample_idx in range(len(self.df.index)):
            for worker_idx in range(2, num_workers):
                list_df.iloc[(sample_idx*num_workers) + worker_idx] = [self.df.iloc[sample_idx, 0], self.df.iloc[sample_idx, 1], worker_idx, self.df.iloc[sample_idx, worker_idx + 2]]

        # remove all rows with missing values for labels from dataframe list.
        list_df = list_df[~list_df.eq(missing_value).any(1)]

        if not inplace:
            return list_df
        self.df = list_df

    def __len__(self):
        """ get size of dataset

        Returns
        -------
        int
            number of rows in dataset. If sparse matrix, number of
            samples.
        """
        return len(self.labels)

    def __getitem__(self, i):
        """ get specific row from dataset

        Returns
        -------
        dict:
            {header: value, header: value, ...}
        """
        #row = self.df.iloc[i]
        #return dict(row)
        # TODO Currently uncertiain what would make the most intuitive sense for
        # this feature, given the multiple dataframes.
        raise NotImplementedError

    def convert_to_annotation_list(self, df=None):
        """Converts from sparse matrix format into annotation list format."""
        # TODO Currently expects df to be in sparse matrix format without a check!
