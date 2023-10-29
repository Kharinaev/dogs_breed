from pathlib import Path
from typing import List


def make_dirs(directories_list: List[str]):
    paths = []
    for directory in directories_list:
        path = Path(directory)
        if not path.exists():
            path.mkdir()
        paths.append(path)
    return paths
