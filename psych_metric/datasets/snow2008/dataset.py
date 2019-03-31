import numpy as np
import os
import pandas as pd
import ast

from psych_metric.datasets.base_dataset import BaseDataset

ROOT = os.environ['ROOT']
HERE = os.path.join(ROOT, 'psych_metric/datasets/snow2008/')

class Snow2008(BaseDataset):
    """class that loads and serves data from Snow 2008 
    
    Attributes
    ----------
    dataset : str
        Name of specific dataset
    df : pandas.DataFrame
        Data Frame containing annotations
    """

    def __init__(self, dataset='anger'):
        """initialize class by loading the data

        Parameters
        ----------
        dataset : str
            the name of one of the subdatasets corresponding to file name
        """
        dsets = [
            'anger', 'disgust', 'fear', 
            'joy', 'rte', 'sadness', 'surprise', 
            'temp', 'valence', 'wordsim', 'wsd'
        ]
        assert dataset in dsets
        self.dataset = dataset
        annotation_file = '{}.standardized.tsv'.format(self.dataset)
        annotation_file = os.path.join(
                HERE, 'rion_snow_2008_simulated_data', annotation_file
        )
        self.df = self.load_tsv(annotation_file)

    def load_tsv(self, f):
        """ Read and parse the published dataset file
        
        Parameters
        ----------
        f : str
            path of tsv file

        Returns
        -------
        pandas.DataFrame
            Data Frame of annotations

        """
        df = pd.read_csv(f, header=0, delimiter='\t')
        df.columns = [c.replace('!', '') for c in df.columns]
        return df

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
