# Research Repository for ND CVRL + PA Joint Metric Research

## Experiments
To run the experiments, run python or ipython from the top level dir of the repository if you want to use the current code from `psych_metric` package, rather than an installed version of the package using `setup.py`.

## Repository Structure

Loosely follows [Cookiecutter](https://drivendata.github.io/cookiecutter-data-science/) structure

`makefile` contains useful commands that are run often. run `make jupyter.port` to open a jupyter notebook to be accessed in a web browser at `localhost:16123`

## Initializing Environment

[INSTALL.md](INSTALL.md)

## Communication

Slack `#metric` channel on ND CVRL Slack


## Reports, Proposals, Ideas

Shared Google Drive Folder

## Final Paper

Sharelatex / Overleaf

## Code and Data

The preferred process is:
1. Create a new branch
2. edit that branch
3. `git pull --rebase` changes from master
4. Make a merge request
5. Discuss changes
6. Merge with master

But given the nature of the project trivial changes to the repository can be pushed to master without that process.

Data should be linked in shared google drive folders, make sure to add data folders to .gitignore before pushing.

After creating something useful, add a simple jupyter notebook in the proper folder as an example of how to use it, e.g. adding a new dataset.
