"""
Base Dataset class to be inherited by other dataset classes
Overwrite these methods
"""
from sklearn.preprocessing import LabelEncoder
import ast
import numpy as np

class BaseDataset(object):

    def split_train_val_test(self):
        raise NotImplementedError

    @staticmethod
    def str_to_array(s):
        if isinstance(s, str):
            s = ast.literal_eval(s)
        return np.array(s)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def display(self, i):
        raise NotImplementedError

    def _check_dataset(self, dataset, dsets):
        """Checks if the dataset is in the provided set. Raises error if not.

        Params
        ------
        dataset : str
            string representing the dataset identity.
        dsets : set
            set of strings that is compared to for evaluating dataset's validity.
        """
        if dataset not in dsets:
            raise ValueError(str(dataset) + ' is not an acceptable dataset. Use only the following datasets: ' + str(dsets))

    def read_csv_to_dict(csvpath, sep=',', key_column=0, value_column=1, key_dtype=str):
        """Read the csv as a dict where key value pair is first two columns."""
        #TODO I do not see this being used often, perhaps make another subclass of BaseDataset, and all that use this extend that class.
        csv_dict = {}
        with open(csvpath, 'w') as f:
            csv_reader = csv.reader(f, delimiter=sep)

            if key_dtype != str:
                for row in csv_reader:
                    csv_dict[key_dtype(row[key_column])] = row[value_column]
            else:
                # No need to cast the row key value.
                for row in csv_reader:
                    csv_dict[row[key_column]] = row[value_column]
        return csv_dict

    def add_ground_truth_to_samples(self, ground_truth, inplace=True, is_dict=False, sample_id='sample_id'):
        """ Add the ground truth labels to every sample (row) of the main
        dataframe; in place by default.

        Parameters
        ----------
        ground_truth : pandas.DataFrame
            Dataframe of the ground truth,
        inpalce : bool, optinal
            Update dataframe in place if True, otherwise return the updated
            dataframe
        sample_id : str
            the column name that identifies the sample to match to its
            corresponding ground truth value.

        Returns
        -------
        pd.DataFrame
            DataFrame with a ground truth labels column returned if inplace is
            False, otherwise returns None.
        """
        if not is_dict:
            # converts to dict first, may or may not be efficent.
            ground_truth_dict = {}
            for i in range(len(ground_truth)):
                ground_truth_dict[ground_truth.iloc[i,0]] = ground_truth.iloc[i,1]
            ground_truth = ground_truth_dict

        ground_truth_col = self.df[sample_id].apply(lambda x: ground_truth[x] if x in ground_truth else None)

        # TODO make this default to not inplace, it's safer that way.
        if inplace:
            self.df['ground_truth'] = ground_truth
        else:
            df_copy = self.df.copy()
            df_copy['ground_truth'] = ground_truth
            return df_copy

    def encode_labels(self, columns=None, column_labels=None, by_index=False):
        """Initializes a label encoder and fits it to the expected labels of
        the data column.

        This creates a label encoder and encodes a column in the dataframe with
        that label encoder.

        Parameters
        ----------
        column : list, optional
            list of str or int objects representing the columns to encode. If
            column is not included in dictionary keys, but is in `columns`, then
            the column will encode but using only the unique values seen from
            the data (will not include missing values).
        column_labels : dict, optional
           dict of column names to target labels for fitting the label encoder
           for that column.
        by_index : bool, optional
            Accesses column by column index if True, by column label if False

        Returns
        -------
        {str : sklearn.preprocessing.LabelEncoder}
            column name to LabelEncoder for the labels to their numeric
            representations.
        """
        label_encoders = dict()

        if columns is None and column_labels is None:
            raise ValueError('need to provide either the columns or column_labels parameters.')

        # if column_labels is provided, encode them
        if column_labels is not None:
            for column in column_labels.keys():
                label_encoder = LabelEncoder()
                label_encoder.fit(list(column_labels[column]))

                if by_index:
                    self.df.iloc[:,column] = label_encoder.transform(self.df.iloc[:,column])
                else:
                    self.df[column] = label_encoder.transform(self.df[column])

                label_encoders[column] = label_encoder

        # if columns is provided, encode them if not already encoded
        if columns is not None:
            if column_labels is not None:
                # Only encode the columns not in column_labels
                columns = set(columns) - set(column_labels.keys())

            for column in columns:
                label_encoder = LabelEncoder()

                if by_index:
                    self.df.iloc[:,column] = label_encoder.fit_transform(self.df.iloc[:,column])
                else:
                    self.df[column] = label_encoder.fit_transform(self.df[column])

                label_encoders[column] = label_encoder

        return label_encoders

    def convert_to_annotation_list(self, missing_value=None, inplace=False):
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

        num_workers = len(self.df.columns)
        list_df = pd.DataFrame(np.empty((len(self.df.index)*num_workers, 3)), columns=['sample_id', 'worker_id', 'worker_label'])

        # Place the correct value in the resulting dataframe list
        for sample_idx in range(len(self.df.index)):
            for worker_idx in range(num_workers):
                list_df.iloc[(sample_idx*num_workers) + worker_idx] = [sample_idx, worker_idx, self.df.iloc[sample_idx, worker_idx]]

        # remove all rows with missing values for labels from dataframe list.
        list_df = list_df[~list_df.eq(missing_value).any(1)]

        if not inplace:
            return list_df
        self.df = list_df

    #def convert_to_sparse_matrix(self, annotations=None):
    #    """Convert provided dataframe into a matrix format, possibly sparse."""
    #    raise NotImplementedError

    def truth_inference_survey_format(self):
        """Creates the two dictionaries the Truth Inference Survey's code
        expects from the dataframe in list format into the two.

        Returns
        -------
            Tuple of a dictionary of sample identifiers as keys and values as list of
            annotator id and their annotation, and a dictionary of annotator
            identifeirs as keys and values as list of samples.
        """
        # NOTE may want this to handle more standard format versions of data.
        samples_to_annotations = dict()
        annotators_to_samples = dict()

        for index, row in self.df.iterrows():
            # add to samples_to_annotations
            if row['sample_id'] not in samples_to_annotations:
                samples_to_annotations[row['sample_id']] = [row['worker_id'], row['worker_label']]
            else:
                samples_to_annotations[row['sample_id']].append([row['worker_id'], row['worker_label']])

            # add to annotators_to_samples
            if row['worker_id'] not in annotators_to_samples:
                annotators_to_samples[row['worker_id']] = [row['sample_id'], row['worker_label']]
            else:
                annotators_to_samples[row['worker_id']].append([row['sample_id'], row['worker_label']])

        return samples_to_annotations, annotators_to_samples
