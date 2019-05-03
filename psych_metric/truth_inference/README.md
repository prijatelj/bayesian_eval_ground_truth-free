#Truth Inference
This directory contains all of the models and methods used for the Truth Inference task, which is the task of inferring truth from potenitally unreliable sources. In our case, those unreliable sources are human annotators' annotations of data. Methods of truth inference are focused on extracting what truth exists within the unreliable sources. In the case where there is no possible truth (no quantifiable actual ground truth), then these methods are typically for compressing the information in the multiple sources and removing or decreasing the affect that malicious or inferior sources have on the aggregated information.

Majority of these methods, specifically the baselines, will be concerned with annotation aggregation. Our concern with regards to truth inference is estimating and evaluating the information contained within the annotators such that we may provide our predictive models and metrics (comparison methods) with the information that is useful from the annotators for performing the desired task. We want the resulting predictive models to learn that task the best they can from this unreliable data source such that they may generalize well and (hopefully) perform better than humans on that task.

##Annotator Aggregation Models
The majority of the baseline annotator aggregation methods are taken from two surveys, "Comparing Bayesian Models of Annotation" by Silviu Paun et. al 2018 and "Truth Inference in Crowdsourcing: Is the Problem Solved?" by Yudian Zheng et. al 2017.

A work very close to our own objective, and not just turth inference via annotator aggregation methods is Crowd Layer from "Deep Learning from Crowds" by Filipe Rodrigues and Francisco C. Pereira in 2018. However, our focus is on a metric that can evaluate a predictive model's performance compared to the annotators, while Crowd Layer simply improves the Artificial Neural Network model it is applied to. At the end of this, our metric should be able to better evaluate Crowd Layer and similar models. I believe crowd layer belongs in the truth inference folder due to being able to extract the weights of the annotators from the crowd layers' weights. 



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
