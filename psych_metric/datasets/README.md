Datasets
==
All datasets and their data handler code are contained within this directory.

Standardization
--
Standardization of the data will be handled via their respective `dataset.py`. This is necessary such that we do not have to code different handlers for every truth inference (annotator aggregator) method.

- df : pandas.DataFrame
    + columns: features, worke\_id, label, ground\_truth an annotation list.
- sample_count
    + int: number of samples within this loaded dataset.
- annotator_count
    + int: number of annotators within this loaded dataset.

ground\_truth and features will have the same number of rows, so they could theorectically be combined to make a `data` dataframe, which is not very clear and will require the features to be extracted from the dataframe when(if) used in training of predictive models. Probably not desirable in the end.

There needs to be some form of standardization to get the feature data from the class if the data is not actually contained within the features dataframe (ie. images, and the sample_id is the filename of that image). If this is the case, it will result in the features dataframe being useless for that dataset, and requiring some function to aid in the loading and further handling of the data.
