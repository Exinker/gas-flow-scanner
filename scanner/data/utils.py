import os
from typing import Literal


def fetch_filedir(kind: Literal['data', 'img']) -> str:
    filedir = os.path.join('.', kind)

    if not os.path.exists(filedir):
        os.mkdir(filedir)

    return filedir
