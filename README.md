[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8.8](https://img.shields.io/badge/python-3.8.8-blue.svg)](https://www.python.org/downloads/release/python-388/)
[![PyTorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.0-red.svg)](https://pytorch.org/blog/pytorch-1.8-released/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-white)](https://pycqa.github.io/isort/)
[![Linting: flake8 & mypy & pylint](https://img.shields.io/badge/linting-flake8%20%26%20mypy%20%26%20pylint-deepblue)](https://pypi.org/project/pytest-pylint/)
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)

# 모두를 위한 메타러닝: PyTorch를 활용한 Few-shot 학습 모델과 빠른 강화학습 에이전트 만들기

This repository contains PyTorch implementations of meta-reinforcement learning algorithms.

## Prerequisites

This repository is implemented and verified on **python 3.8.8**

## Installation

To run on **pytorch 1.8.1**, enter the [pytorch version link](https://pytorch.org/get-started/previous-versions/#wheel) and run the installation command to desired specifications.

Next, clone this repository and run the following command.

```shell
$ make setup
```

## Usages

The repository's high-level structure is:

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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/dongminlee94/"><img src="https://avatars.githubusercontent.com/u/29733842?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dongmin Lee</b></sub></a><br /><a href="https://github.com/dongminlee94/meta-rl/commits?author=dongminlee94" title="Code">💻</a> <a href="https://github.com/dongminlee94/meta-rl/commits?author=dongminlee94" title="Documentation">📖</a</td>
    <td align="center"><a href="https://github.com/Clyde21c/"><img src="https://avatars.githubusercontent.com/u/35162035?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Seunghyun Lee</b></sub></a><br /><a href="https://github.com/dongminlee94/meta-rl/commits?author=Clyde21c" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
