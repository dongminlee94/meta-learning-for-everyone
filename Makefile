GPU := $(shell which nvcc)
ifdef GPU
    DEVICE="gpu"
else
    DEVICE="cpu"
endif

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
	pip install -U setuptools
	pip install -e .
	pip install -r requirements-common.txt
	pip install -r requirements-$(DEVICE).txt
	bash ./hooks/install.sh

init-dev:
	make init
	pip install -r requirements-dev.txt
