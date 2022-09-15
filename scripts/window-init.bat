pip install poetry==1.1.15
pip install -U pip
poetry install
pre-commit install
python3 ./scripts/download-torch.py
conda install -y tensorboard
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
python -m ipykernel install --user
