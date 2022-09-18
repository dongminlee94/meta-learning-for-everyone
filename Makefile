all: init format lint

format:
	black . --line-length 104
	isort .
	nbqa black .
	nbqa isort .

lint:
	pytest src/ --pylint --flake8 --ignore=src/meta_rl/envs

init:
	pip install -U pip
	pip install -e .
	pip install -r requirements.txt
	python3 ./scripts/download-torch.py
	conda install -y tensorboard
	jupyter contrib nbextension install --user
	jupyter nbextensions_configurator enable --user
	python3 -m ipykernel install --user

init-dev:
	make init
	pip install -r requirements-dev.txt
	bash ./scripts/install.sh
