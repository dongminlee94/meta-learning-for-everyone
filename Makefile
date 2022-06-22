STAGED := $(shell git diff --cached --name-only --diff-filter=ACMR -- 'src/***.py' | sed 's| |\\ |g')

all: format lint
	echo 'Makefile for meta-learning-for-everyone repository'

format:
	black .
	isort .
	nbqa black .
	nbqa isort .

lint:
	pytest src/ --pylint --flake8 --ignore=src/meta_rl/envs
	# nbqa pytest src/ --pylint --flake8

lint-all:
	pytest src/ --pylint --flake8 --ignore=src/meta_rl/envs --cache-clear
	# nbqa pytest src/ --pylint --flake8 --cache-clear

lint-staged:
ifdef STAGED
	pytest $(STAGED) --pylint --flake8 --ignore=src/meta_rl/envs --cache-clear
	# nbqa pytest $(STAGED) --pylint --flake8 --cache-clear
else
	@echo "No Staged Python File in the src folder"
endif

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
