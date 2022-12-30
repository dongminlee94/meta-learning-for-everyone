# -*- coding: utf-8 -*-
import os
import sys

import GPUtil


def main() -> None:
    cmd = "pip install torch==1.9.1"

    if sys.platform == "win32" or sys.platform == "linux":
        if GPUtil.getAvailable():
            cmd += "+cu111 -f https://download.pytorch.org/whl/torch_stable.html"
        else:
            cmd += "+cpu -f https://download.pytorch.org/whl/torch_stable.html"
    elif sys.platform == "darwin":
        cmd += " -f https://download.pytorch.org/whl/torch_stable.html"

    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
