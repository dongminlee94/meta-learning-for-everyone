[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/dongminlee94/meta-learning-for-everyone.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/dongminlee94/meta-learning-for-everyone/context:python)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8.8](https://img.shields.io/badge/python-3.8.8-blue.svg)](https://www.python.org/downloads/release/python-388/)
[![PyTorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.1-red.svg)](https://pytorch.org/blog/pytorch-1.8-released/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-white)](https://pycqa.github.io/isort/)
[![Linting: flake8 & mypy & pylint](https://img.shields.io/badge/linting-flake8%20%26%20mypy%20%26%20pylint-deepblue)](https://pypi.org/project/pytest-pylint/)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# Meta Learning for Everyone: Developing Few-shot Learning Models and Fast Reinforcement Learning Agents using PyTorch

This repository is a repository for the book "Meta-Learning for Everyone".

## Requirements

This repository is implemented and verified on **python 3.8.8**.

## Installation and Usage

### 1. Install Anaconda

First, install Anaconda from the link below.

https://www.anaconda.com/

### 2. Create Anaconda envrionment

Second, follow the commands below to create a new python environment and activate the created environment.

```bash
(base) $ conda create -y -n meta python=3.8.8

(base) $ conda activate meta

(meta) $ conda env list
```

### 3. Install packages

Next, after cloning this repository, run the following command to install the required packages.

**MacOS & Linux user**

```bash
# User
(meta) $ make init

# Developer
(meta) $ make init-dev
```

**Windows user**

```bash
# User
(meta) $ "./scripts/window-init.bat"
```

### 4. Train models & Check results

**Meta-SL**

For Meta-SL, move to each algorithm folder, run the algorithms using `jupyter notebook`, and check the results.

```bash
(meta) $ jupyter notebook
```

If you are trying to use Colab, please install PyTorch-related packages by executing the command bellow.

```python
!pip install torchmeta torchtext==0.10.1 torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**Meta-RL**

For Meta-RL, move to each algorithm folder and run the commands below.

```bash
# RL^2
(meta) $ python rl2_trainer.py

# MAML
(meta) $ python maml_trainer.py

# PEARL
(meta) $ python pearl_trainer.py
```

In the case of Meta-RL, please run the Tensorboard command below to check the results of the meta-training and meta-testing you executed.

```bash
(meta) $ tensorboard --logdir=./results
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/dongminlee94/"><img src="https://avatars.githubusercontent.com/u/29733842?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dongmin Lee</b></sub></a><br /><a href="https://github.com/dongminlee94/meta-learning-for-everyone/commits?author=dongminlee94" title="Code">💻</a> <a href="https://github.com/dongminlee94/meta-learning-for-everyone/commits?author=dongminlee94" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/Clyde21c/"><img src="https://avatars.githubusercontent.com/u/35162035?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Seunghyun Lee</b></sub></a><br /><a href="https://github.com/dongminlee94/meta-learning-for-everyone/commits?author=Clyde21c" title="Code">💻</a> <a href="https://github.com/dongminlee94/meta-learning-for-everyone/commits?author=Clyde21c" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/LunaJang"><img src="https://avatars.githubusercontent.com/u/25239851?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Luna Jang</b></sub></a><br /><a href="https://github.com/dongminlee94/meta-learning-for-everyone/commits?author=LunaJang" title="Code">💻</a></td>
    <td align="center"><a href="https://www.endtoend.ai"><img src="https://avatars.githubusercontent.com/u/6107926?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Seungjae Ryan Lee</b></sub></a><br /><a href="https://github.com/dongminlee94/meta-learning-for-everyone/commits?author=seungjaeryanlee" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
