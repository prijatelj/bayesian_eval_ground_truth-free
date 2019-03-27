"""
Base Dataset class to be inherited by other dataset classes
Overwrite these methods
"""

class BaseDataset(object):

    def split_train_val_test(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, i):
        raise NotImplementedError

    def display(self, i):
        raise NotImplementedError
