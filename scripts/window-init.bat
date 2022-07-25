pip install -e .
pip install -r requirements.txt
python ./scripts/download-torch.py
conda install -y tensorboard
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
python -m ipykernel install --user
