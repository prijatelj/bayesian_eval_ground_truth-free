# TODO this needs handled in the __init__ files/setup.py installation of the package.
from psych_metric import datasets

def dataset_exists(dataset, collection=None)
    """Returns true if the provided string identifier matches a dataset in
    some collections.
    """
    if collection is None:
        # checks if the dataset exists in any collection
        return isinstance(dataset, str) and (dataset in datasets.Snow2017.datasets or dataset in datasets.TruthSurvey2017.datasets or dataset == 'trec-fr10-data')
    elif collection == 'truth_survey_2017':
        # Check if dataset in truth_survey_2017
        return dataset in datasets.TruthSurvey2017.datasets
    elif collection == 'snow_2008':
        # Check if dataset in snow_2008
        return dataset in datasets.Snow2017.datasets

    return dataset == 'trec-rf10-data'

def load_dataset(dataset, encode_columns=False):
    """
    Handler function for calling the correct datahandler class for a given
    dataset.

    Parameters
    ----------
    dataset : str
        name of dataset represented as a string.

    Returns
    -------
    psych_metric.dataset.base_dataset
        A dataset object corresponding to the given dataset str requested.
    """
    #Snow 2018
    if DataHandler.exists(dataset, 'snow_2008'):
        return datasets.Snow2018(dataset)

    # Truth Survey 2017
    elif DataHandler.exists(dataset, 'truth_survey_2017'):
        return datasets.TruthSurvey2017(dataset, encode_columns)

    # TREC 2010
    elif DataHandler.exists(dataset, 'trec-fr10'):
        return datasets.TRECRelevancy2010(dataset, encode_columns)
