import numpy as np
import os
import pandas as pd
import ast

from psych_metric.datasets.base_dataset import BaseDataset

ROOT = os.environ['ROOT']
HERE = os.path.join(ROOT, 'psych_metric/datasets/pa/')

class PA(BaseDataset):
    """class that loads and serves data from Perceptive Automata
    
    Attributes
    ----------
    dataset : str
        Name of specific dataset
    df : pandas.DataFrame
        Data Frame containing annotations
    """

    def __init__(self, dataset='3', date='032818'):
        """initialize class by loading the data

        Parameters
        ----------
        dataset : int or str
            experiment id of dataset
        date : str
            date data was exported
        """
        dataset = str(dataset)
        dsets = ['3', '4']
        assert dataset in dsets
        self.dataset = dataset
        self.date = date
        annotation_file = '{}_{}.csv'.format(self.date, self.dataset)
        annotation_file = os.path.join(
                HERE, 'pa_data', annotation_file
        )
        self.df = self.load_csv(annotation_file)
        self.df = self.set_multinomials(self.df, vote_col='score_array')

    def load_csv(self, f):
        """ Read and parse the dataset file
        
        Parameters
        ----------
        f : str
            path of csv file

        Returns
        -------
        pandas.DataFrame
            Data Frame of annotations

        """
        df = pd.read_csv(f, header=0)
        return df

    def set_multinomials(self, df, vote_col='score_array'):
        def get_hist(votes):
            if isinstance(votes, str): 
                votes = ast.literal_eval(votes)
            bins = range(1, 7)
            return np.histogram(votes, bins=bins)[0]

        df['multinomial'] = df[vote_col].map(get_hist)
        return df

    def get_multinomial_array(self):
        return np.stack(self.df['multinomial'], axis=0)

    def __len__(self):
        """ get size of dataset

        Returns
        -------
        int
            number of annotations in dataset
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
