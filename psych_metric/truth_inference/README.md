#Truth Inference
This directory contains all of the models and methods used for the Truth Inference task, which is the task of inferring truth from potenitally unreliable sources. In our case, those unreliable sources are human annotators' annotations of data. Methods of truth inference are focused on extracting what truth exists within the unreliable sources. In the case where there is no possible truth (no quantifiable actual ground truth), then these methods are typically for compressing the information in the multiple sources and removing or decreasing the affect that malicious or inferior sources have on the aggregated information.

Majority of these methods, specifically the baselines, will be concerned with annotation aggregation. Our concern with regards to truth inference is estimating and evaluating the information contained within the annotators such that we may provide our predictive models and metrics (comparison methods) with the information that is useful from the annotators for performing the desired task. We want the resulting predictive models to learn that task the best they can from this unreliable data source such that they may generalize well and (hopefully) perform better than humans on that task.

##Annotator Aggregation Models
The majority of the baseline annotator aggregation methods are taken from two surveys, "Comparing Bayesian Models of Annotation" by Silviu Paun et. al 2018 and "Truth Inference in Crowdsourcing: Is the Problem Solved?" by Yudian Zheng et. al 2017.

A work very close to our own objective, and not just turth inference via annotator aggregation methods is Crowd Layer from "Deep Learning from Crowds" by Filipe Rodrigues and Francisco C. Pereira in 2018. However, our focus is on a metric that can evaluate a predictive model's performance compared to the annotators, while Crowd Layer simply improves the Artificial Neural Network model it is applied to. At the end of this, our metric should be able to better evaluate Crowd Layer and similar models. I believe crowd layer belongs in the truth inference folder due to being able to extract the weights of the annotators from the crowd layers' weights. 


##Generic Truth Inference Model:
The following is the mathematical formulation of a generic truth inference model. NOTE: use the long form names for the variables, they are just placeholders to make it easier to write the equation. We will use descriptive variable names.

For all a(Y, X, G, alpha, random\_state) in A, where A is the set of all possible Truth Inference models:
- Y : Annotators' Labels of data samples. This is able to be thought of as (sparse) matrix of |samples|,|annotators|, but is more computationally efficient to represent as list of annoations where each element of the list contains the unique identifier of the sample, the specific annotator's identifier who annotated this sample, and that annotator's label for this sample. This could also include the time in the case an annotator annotates multiple samples to provide a sequential nature of the labeling.
- X : (optional, however this work will argue it is ideal to include for truth inference) The data samples that are annotated. This will not be included in some truth inference models because they only focus on the annotations themselves, and as such they could be annotating anything. This is absolutely necessary for any truth inference model that leverages the data.
- G : (optional and more practical to be missing in some cases when it does not exist in practice) The ground truth of the annotated data.
- alpha\* : the set of (hyper)parameters needed by the specific instance of the generic truth inference model. Some of these are similar to optimizers and they require parameters. This is akin to pythons args\* where this set is interpretable only to the specific type of truth inference model.
- (optional) If this model type uses randomization in any part of its truth inference process, then always include `random_state` as a parameter of the truth inference model when it comes to maximizing the ability to reproduce the model's truth inference.

For efficiency and consolidation, all standardized datasets will be represented as a single `pandas.Dataframe` of the format: sample features/id, worker id, ground truth.
This is more pertinent to the data handler code, because these truth inference models will always expect them as these separate values, as is the norm in most standard data science packages/APIs.

##Models
The models of truth inference included in this project.

TODO: get the full names and breif citations so each are identifiable.

###Comparision of Bayesian Models of Annotation
- Majority Vote
- dawid\_skene : Dawid and Skene
- hier\_dawid\_skene : Hierarchial Dawid and Skene
- multinomial
- ItemDiff : Item Difficulty, extension of "Beta-Binomial by Item" Carpenter 2008
- LogRndEff : Logistic Random Effects, Carpenter 2008
- MACE : Multi-Annotator Competence Estimation, Hovy et al. 2013.

###Truth Inference Survey 2017
- Majority Vote
- Mean
- Median
- dawid\_skene : Dawid and Skene (Should include the hierarchial D and S too, perhaps)
- GLAD : Generative model of Labels, Abilities, and Difficulties)
- Minimax : 
- BCC : 
- CBCC : 
- LFC : 
- LFC-N : 
- ZenCrowd : 
- CATD : 
- PM : 
- Multi : "The Multidimensional Wisdom of Crowds" Welinder et al. 2010
- KOS :
- VI-BP : 
- VI-MF :

### Clustering approach paper
- Majority Vote
- spectral\_dawid\_skene
- GTIC : Ground Truth Inference using Clustering
- ZenCrowd : 
