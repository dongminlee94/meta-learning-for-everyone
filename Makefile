format:
		black .
		isort .

lint:
		env PYTHONPATH=rl2 pytest --pylint --flake8 --mypy --ignore=rl2/envs

setup:
		pip install -r requirements.txt
		pre-commit install