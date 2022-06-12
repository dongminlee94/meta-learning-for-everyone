# -*- coding: utf-8 -*-

REPO_ROOT=$(git rev-parse --show-toplevel)

# Install pre-commit
rm -f $REPO_ROOT/.git/hooks/pre-commit && rm -f $REPO_ROOT/.git/hooks/pre-commit.legacy
pip install pre-commit
cd $REPO_ROOT && pre-commit install
