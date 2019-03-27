# common run commands

set.environment:
	source set_environment.sh

jupyter:
	jupyter nbextension enable --py widgetsnbextension
	jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000

jupyter.port:
	jupyter nbextension enable --py widgetsnbextension
	jupyter notebook --no-browser  --NotebookApp.iopub_data_rate_limit=10000000  --port=16123
