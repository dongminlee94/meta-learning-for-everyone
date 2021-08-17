format:
		black . --line-length 104
		isort .

lint:
		env PYTHONPATH=src/rl2 pytest src/rl2 --pylint --flake8 --mypy --ignore=src/rl2/envs
		env PYTHONPATH=src/pearl pytest src/pearl --pylint --flake8 --mypy --ignore=src/pearl/envs

setup:
		pip install -r requirements.txt
		pre-commit install