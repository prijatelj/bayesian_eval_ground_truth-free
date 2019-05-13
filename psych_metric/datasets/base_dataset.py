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

        if inplace:
            self.df['ground_truth'] = ground_truth
        else:
            df_copy = self.df.copy()
            df_copy['ground_truth'] = ground_truth

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

    def convert_to_annotation_list(self, annotations=None):
        """Convert provided dataframe into a annotation list equivalent.

        Converts the given dataframe of dataset's expected onload format into
        a dataframe of equivalent data, but in annotation list format where the
        rows are the different instance of annotations by individual annotators
        and the columns are 'sample_id', 'worker_id', and 'label'.

        Parameters
        ----------
        annotations : pandas.DataFrame
            Dataframe to be converted into an annotation list format. Defaults
            to class's df.

        Returns
        -------
        pandas.DataFrame
            Data Frame of annotations in an annotation list format.

        """
        raise NotImplementedError
