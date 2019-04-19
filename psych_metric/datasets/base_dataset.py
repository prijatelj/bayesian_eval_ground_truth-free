"""
Base Dataset class to be inherited by other dataset classes
Overwrite these methods
"""
from sklearn.preprocessing import LabelEncoder

class BaseDataset(object):

    def split_train_val_test(self):
        #TODO Why have this in the data handler? why not in an evaluation code?
        # OR, in the base dataset code [here], assuming standardization.
        # If no standardization, it makes sense to make it dataset specific.
        raise NotImplementedError

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
