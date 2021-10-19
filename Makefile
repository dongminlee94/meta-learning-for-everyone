format:
		black . --line-length 104
		isort .

lint:
		env PYTHONPATH=src/rl2 pytest src/rl2 --pylint --flake8 --mypy
		env PYTHONPATH=src/maml pytest src/maml --pylint --flake8 --mypy
		env PYTHONPATH=src/pearl pytest src/pearl --pylint --flake8 --mypy

setup:
		pip install -r requirements.txt
		pre-commit install