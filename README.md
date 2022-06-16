[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8.8](https://img.shields.io/badge/python-3.8.8-blue.svg)](https://www.python.org/downloads/release/python-388/)
[![PyTorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.0-red.svg)](https://pytorch.org/blog/pytorch-1.8-released/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-white)](https://pycqa.github.io/isort/)
[![Linting: flake8 & mypy & pylint](https://img.shields.io/badge/linting-flake8%20%26%20mypy%20%26%20pylint-deepblue)](https://pypi.org/project/pytest-pylint/)
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)

# 모두를 위한 메타러닝: PyTorch를 활용한 Few-shot 학습 모델과 빠른 강화학습 에이전트 만들기

"모두를 위한 메타러닝" 책에 대한 코드 레포지토리입니다.

## 필요 조건

이 레포지토리에서는 **python 3.8.8** 버전을 사용합니다.

## 설치 및 사용 방법

먼저, 아래의 링크에서 Anaconda를 설치합니다.
https://www.anaconda.com/

다음으로, 아래의 명령어들을 통해 새로운 python 환경을 만들고, 그 환경을 활성화합니다.

```bash
(base) $ conda create -y -n meta python=3.8.8

(base) $ conda activate meta

(meta) $ conda env list
```

이어서, 이 레포지토리를 clone한 뒤, 다음의 명령어를 실행하여 필요한 패키지들을 설치해주세요.

```bash
$ cd meta-learning-for-everyone

# 사용자
$ make init

# 개발자
$ make init-dev
```

이제 각 알고리즘들을 실행하기 위해 Meta-SL은 `src/meta_sl`으로 이동하고, Meta-RL은 `src/meta_rl`으로 이동해주세요.

```bash
$ cd src/meta_sl

# or

$ cd src/meta_rl
```

Meta-SL은 각 알고리즘 폴더로 가서 `jupyter notebook`을 이용하여 해당 알고리즘을 실행해주세요.

```bash
$ jupyter notebook
```

Meta-RL은 각 알고리즘 폴더로 가서 아래의 명령어들을 이용하여 실행해주세요.

```bash
$ rl2_trainer.py

# or

$ maml_trainer.py

# or

$ pearl_trainer.py
```

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
