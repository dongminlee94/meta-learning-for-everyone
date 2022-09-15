all: init format lint

format:
	poetry run black .
	poetry run isort .
	poetry run nbqa black .
	poetry run nbqa isort .

lint:
	poetry run pytest src/ --pylint --flake8 --ignore=src/meta_rl/envs

init:
	pip install poetry==1.1.15
	pip install -U pip
	poetry install
	pre-commit install
	python3 ./scripts/download-torch.py
	conda install -y tensorboard
	jupyter contrib nbextension install --user
	jupyter nbextensions_configurator enable --user
	python3 -m ipykernel install --user

publish:
	poetry export --dev -f requirements.txt --output requirements.txt --without-hashes
