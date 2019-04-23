# NOTE I was considering Enums here, but i decided to go without them for now.
# Enums for possibly the dataset classes, or their data subsets.

# TODO this needs handled in the __init__ files/setup.py installation of the package.
from psych_metric import datasets

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
    psychmetric.dataset.base_dataset
        A dataset object corresponding to the given dataset str requested.
    """
    # TODO this getting sub dataset clearly needs handled better here ...
    # Only works if all sub datasets names are unique out of ALL datasets.
    snow_2008 = [
        'anger', 'disgust', 'fear',
        'joy', 'rte', 'sadness', 'surprise',
        'temp', 'valence', 'wordsim', 'wsd'
    ]
    truth_survey_2017= [
        'd_Duck Identification', 'd_jn-product', 'd_sentiment',
        's4_Dog data', 's4_Face Sentiment Identification', 's4_Relevance',
        's5_AdultContent', 'f201_Emotion_FULL'
    ]

    #Snow 2018
    if dataset in snow_2008:
        return datasets.Snow2018(dataset)

    # Truth Survey 2017
    elif dataset in truth_survey_2017:
        return datasets.TruthSurvey2017(dataset, encode_columns)

    elif dataset == 'trec-rf10-data':
        return datasets.TRECRelevancy2010(dataset, encode_columns)
