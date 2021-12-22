# A Bayesian Evaluation Framework for Subjectively Annotated Visual Recognition Tasks

This is the research repository for the [Pattern Recognition paper](https://www.sciencedirect.com/science/article/pii/S0031320321005604) completed as a joint project by the University of Notre Dame's Computer Vision and Research Lab and Perceptive Automata.

This work uses Tensorflow and Tensorflow Probability (TFP) prior to the Tensorflow v2 release.
TFP is used for the Bayesian Evaluator models.
This includes the distributions and the HMC Bayesian Neural Network (BNN).

## Directory Structure

When installing out code, we recommend using a virtual environment, such as venv or conda.
The models used in experimentation are contained within `psych_metric` and the experiments are contained under `experiments`.

Note that use of this repo for the Bayesian Evaluators requires installing via setup.py for the main package and `setup_exp.py` for the experiments in your python virtual environment.
Any other files are unnecessary for that portion of the project, as well as for the LabelMe and SCUT-FB5500 predictors.

## Experiments
To run the experiments, run python or ipython from the top level dir of the repository if you want to use the current code from `psych_metric` package, rather than an installed version of the package using `setup.py`.

Install using `python setup.py install`

The code and scripts for the experiments are contained within `experiments`

### License

Our code contributions within this repository are released under the MIT License located in `LICENSE.txt`

### Citations

If you use our work, please use the following Bibtex to cite our paper:

```
@article{prijatelj_bayesian_2021,
	title = {A {Bayesian} {Evaluation} {Framework} for {Subjectively} {Annotated} {Visual} {Recognition} {Tasks}},
	issn = {0031-3203},
	url = {https://www.sciencedirect.com/science/article/pii/S0031320321005604},
	doi = {10.1016/j.patcog.2021.108395},
	language = {en},
	urldate = {2021-11-08},
	journal = {Pattern Recognition},
	author = {Prijatelj, Derek S. and McCurrie, Mel and Anthony, Samuel E. and Scheirer, Walter J.},
	month = {oct},
	year = {2021},
	keywords = {Bayesian inference, Bayesian modeling, Epistemic uncertainty, mine, Supervised learning, Uncertainty estimation},
	pages = {108395},
}
```

### Verisoning

This project uses [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
This project's version will remain < 1.0.0 until adequate unit test coverage exists.
