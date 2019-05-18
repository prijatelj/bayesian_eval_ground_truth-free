Datasets
==
All datasets and their data handler code are contained within this directory.

Standardization
--
Standardization of the data will be handled via their respective `dataset.py`. This is necessary such that we do not have to code different handlers for every truth inference (annotator aggregator) method.

- df : pandas.DataFrame
    + The dataframe containing the data of this dataset. It can be in various formats.
    + Annotaiton List format: columns: sample\_id, worker\_id, label, ground\_truth an annotation list.
- sample\_count
    + int: number of samples within this loaded dataset.
- annotator\_count
    + int: number of annotators within this loaded dataset.

ground\_truth and features will have the same number of rows, so they could theorectically be combined to make a `data` dataframe, which is not very clear and will require the features to be extracted from the dataframe when(if) used in training of predictive models. Probably not desirable in the end.

There needs to be some form of standardization to get the feature data from the class if the data is not actually contained within the features dataframe (ie. images, and the sample\_id is the filename of that image). If this is the case, it will result in the features dataframe being useless for that dataset, and requiring some function to aid in the loading and further handling of the data.

TODO
--
- Data loaders and handlers for each dataset. (The actual data, not just the annotations, curently only the annotations are handled in a standardized manner).
- (Perhaps) consider creating simulations of annotations as MIG-MAX did, but able to simulate many different dependencies/causal structures of the annotators. Possibly also simulating the reaction time and other psychometrics.
