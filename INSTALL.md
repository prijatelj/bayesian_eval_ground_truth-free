# Setting up the environment

### Anaconda

[Download](https://www.anaconda.com/distribution/)

Create an environment from a `yaml` file with `conda create -f <filename>`

Create a new environment from scratch with `conda create --name metric_py3 python=3.6`

Add your conda environment to jupyter with: 

```bash
python -m ipykernel install --user --name metric_py3 --display-name "metric_py3"
```

### Environment Variables

Set environment variables in `set_environment.sh` and run `source set_environment.sh`

Access those environment variables in python with `os.environ['ENV_VARIABLE']`

Hopefully the only use case for this is for specifying data paths relative to the repo's ROOT folder (`os.environ['ROOT']`)

### Using the repository as a module

Running `source set_environment.sh` will add the repository to your `PYTHONPATH` so you can run `import psych_metric.psychmetric.<module>` to access code

Simply ensure any file you want to be imported exists in a folder with an `__init__.py` file

TODO: alternatively we can use `setup.py` with `pip install -e`
