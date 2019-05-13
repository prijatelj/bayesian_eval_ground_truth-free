# TODO this needs handled in the __init__ files/setup.py installation of the package.
from psych_metric import datasets

def dataset_exists(dataset, collection=None)
    """Returns true if the provided string identifier matches a dataset in
    some collections.
    """
    if collection is None:
        # checks if the dataset exists in any collection
        return isinstance(dataset, str) and (dataset in datasets.Snow2017.datasets or dataset in datasets.TruthSurvey2017.datasets or dataset == 'trec-fr10-data', or dataset in datasets.Ipeierotis2010.datasets or dataset in datasets.FacialBeauty.datasets or dataset in datasets.CrowdLayer.datasets)
    elif collection == 'truth_survey_2017':
        # Check if dataset in truth_survey_2017
        return dataset in datasets.TruthSurvey2017.datasets
    elif collection == 'snow_2008':
        # Check if dataset in snow_2008
        return dataset in datasets.Snow2017.datasets
    elif collection == 'ipierotis_2010':
        return dataset in datasets.Ipierotis2010.datasets
    elif collection == 'facial_beauty':
        return dataset in datasets.FacialBeauty.datasets
    elif collection == 'crowd_layer':
        return dataset in datasets.CrowdLayer.datasets

    return dataset == 'trec-rf10-data'

def load_dataset(dataset, dataset_filepath=None, encode_columns=False):
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
        return datasets.Snow2018(dataset, dataset_filepath)

    # Truth Survey 2017
    elif DataHandler.exists(dataset, 'truth_survey_2017'):
        return datasets.TruthSurvey2017(dataset, dataset_filepath, encode_columns)

    # TREC 2010
    elif DataHandler.exists(dataset, 'trec-fr10'):
        return datasets.TRECRelevancy2010(dataset_filepath, encode_columns)

    # TODO Ipierotis
    elif DataHandler.exists(dataset, 'ipierotis_2010'):
        return datasets.Ipierotis2010(dataset, dataset_filepath, encode_columns)

    # TODO Facial Beauty
    elif DataHandler.exists(dataset, 'facial_beauty'):
        return datasets.FacialBeauty(dataset, dataset_filepath, encode_columns)

    # TODO First Impressions

    # TODO CrowdLayer
    elif DataHandler.exists(dataset, 'crowd_layer'):
        return datasets.CrowdLayer(dataset, dataset_filepath, encode_columns)

    # TODO Google fact evaluation, low priority.
