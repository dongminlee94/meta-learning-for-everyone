[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/dongminlee94/meta-learning-for-everyone.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/dongminlee94/meta-learning-for-everyone/context:python)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8.8](https://img.shields.io/badge/python-3.8.8-blue.svg)](https://www.python.org/downloads/release/python-388/)
[![PyTorch 1.9.1](https://img.shields.io/badge/pytorch-1.9.1-red.svg)](https://pytorch.org/blog/pytorch-1.9-released/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-white)](https://pycqa.github.io/isort/)
[![Linting: flake8 & pylint](https://img.shields.io/badge/linting-flake8%20%26%20pylint-deepblue)](https://pypi.org/project/pytest-pylint/)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<p align="center">
  <img src='img/cover.jpeg' width="400" />
</p>

**This repository supports English. If you use English, move to [language/english](https://github.com/dongminlee94/meta-learning-for-everyone/tree/language/english) branch.**

# 모두를 위한 메타러닝: PyTorch를 활용한 Few-shot 학습 모델과 빠른 강화학습 에이전트 만들기

"모두를 위한 메타러닝" 책에 대한 코드 레포지토리입니다.

## 필요 조건

이 레포지토리에서는 **python 3.8.8** 버전을 사용합니다.

## 설치 및 사용 방법

### 1. Anaconda 설치

먼저, 아래의 링크에서 Anaconda를 설치합니다.

https://www.anaconda.com/

### 2. Anaconda 환경 만들기

다음으로, 아래의 명령어들을 통해 새로운 python 환경을 만들고, 그 환경을 활성화합니다.

```bash
(base) $ conda create -y -n meta python=3.8.8

(base) $ conda activate meta

(meta) $ conda env list
```

### 3. 패키지 설치

이어서, 이 레포지토리를 clone한 뒤, 다음의 명령어를 실행하여 필요한 패키지들을 설치해주세요.

**MacOS 및 Linux 사용자**

```bash
# 사용자
(meta) $ make init

# 개발자
(meta) $ make init-dev
```

**Windows 사용자**

```bash
# 사용자
(meta) $ "./scripts/window-init.bat"
```

### 4. 모델 학습 및 결과 확인

**Meta-SL**

Meta-SL은 각 알고리즘 폴더로 이동하여 `jupyter notebook`을 이용하여 해당 알고리즘을 실행해주시고 결과를 확인해주세요.

```bash
(meta) $ jupyter notebook
```

Colab을 이용하실 경우, 아래의 명령어를 cell에 입력하여 PyTorch 관련 패키지들을 설치하고 이용해주세요.

```python
!pip install torchmeta torchtext==0.10.1 torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**Meta-RL**

Meta-RL은 각 알고리즘 폴더로 이동하여 아래의 명령어들을 이용하여 실행해주세요.

```bash
# RL^2
(meta) $ python rl2_trainer.py

# MAML
(meta) $ python maml_trainer.py

# PEARL
(meta) $ python pearl_trainer.py
```

Meta-RL의 경우, 텐서보드를 이용하여 학습 결과를 확인해주세요.

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
