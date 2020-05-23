# Data Description
"Deep Learning from Crowds" 2018

LabelMe : Multi-class classification; annotations from Amazon Mechanical Turk
    Image Classification of scenes ; what type of location they originate from.
    This has _actual_ ground truth.
    1000 samples annotated by average 2.547 Amazon Mechanical Turk workers.
    59 unique annotators (had to remove no info annotators)
    With uninformative annotators (those who vote 1 symbol only)
        Annotations per annotator:
            mean: 43.169
                       0 .25 .5 .75  1
            quantiles: 3, 9, 27, 66, 182
        Annotator per sample:
            mean: 10.547
                       0 .25  .5  .75 1
            quantiles: 9, 10, 11, 11, 11

    8 classes:
        highway, inside city, tall building, street, forest, coast, mountain or open country

MovieReviews : regression; annotations from Amazon Mechanical Turk
    Text Regression : scale [1, 10]
    1500 samples (movie reviews)
    137 total workers.
    mean 4.96 annotators per sample

2003 CONLL NER task : sequence labelling; annotatios from Amazon MEchanical Turk
    Named Entity Recognition
    5985 labelled samples
    47 total workers

# Download
Download `crowd_layer_data` directory from google drive and uncompress its contents into `crowd_layer_data`.`
