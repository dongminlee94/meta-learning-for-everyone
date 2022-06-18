# -*- coding: utf-8 -*-
import os
import sys

import GPUtil


def main() -> None:
    if sys.platform == "win32" or sys.platform == "linux":
        if GPUtil.getAvailable():
            cli = "pip install torch==1.8.1+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html"
        else:
            cli = "pip install torch==1.8.1+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
    elif sys.platform == "darwin":
        cli = "pip install torch==1.8.1 torchvision==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html"
    print(cli)
    os.system(cli)


if __name__ == "__main__":
    main()
