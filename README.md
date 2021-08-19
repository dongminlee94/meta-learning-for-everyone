<div align="center">
  <br>
  <img src="./assets/meta-rl.png" width="450">
</div>

<p style='text-align: right;'> https://cs330.stanford.edu/slides/cs330_lifelonglearning_karol.pdf </p>

<br>

[![Python 3.8.8](https://img.shields.io/badge/python-3.8.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-white)](https://pycqa.github.io/isort/)
[![Linting: flake8 & mypy & pylint](https://img.shields.io/badge/linting-flake8%20%26%20mypy%20%26%20pylint-green)](https://pypi.org/project/pytest-pylint/)

# Meta-Reinforcement Learning Algorithms with PyTorch

This repository contains PyTorch implementations of meta-reinforcement learning algorithms.

## Prerequisites

This repository is implemented and verified on **python 3.8.8**

## Installation

To run on **pytorch 1.8.0**, enter the [pytorch version link](https://pytorch.org/get-started/previous-versions/#wheel) and run the installation command to desired specifications.

Next, clone this repository and run the following command.

```shell
$ make setup
```

## Python Path

To set python path, move to `meta-rl/`.

```shell
$ cd meta-rl
```

If setting python path on `bashrc`:

```shell
$ echo "export META_HOME=$(pwd)" >> ~/.bashrc
$ echo 'export PYTHONPATH=$META_HOME:$PYTHONPATH' >> ~/.bashrc
```

If setting python path on `zshrc`:

```shell
$ echo "export META_HOME=$(pwd)" >> ~/.zshrc
$ echo 'export PYTHONPATH=$META_HOME:$PYTHONPATH' >> ~/.zshrc
```

## Usages

The repository's high-level structure is:

    └── meta-rl
        └── src
            ├── envs
            ├── rl2
                ├── algorithm
                ├── configs
                └── results
            ├── maml
                ├── algorithm
                ├── configs
                └── results
            └── pearl
                ├── algorithm
                ├── configs
                └── results

### RL^2

TBU

### MAML

TBU

### PEARL

TBU

### Development

We have setup automatic formatters and linters for this repository.

To run the formatters:

```shell
$ make format
```

To run the linters:

```shell
$ make lint
```

New code should pass the formatters and the linters before being submitted as a PR.