
__all__ = ['format_uniform_path']

import os


def format_uniform_path(path):
    """format the path to a new path which seperated by os.sep.
    """
    path = path.replace("//", os.sep)
    path = path.replace("/", os.sep)
    path = path.replace("\\", os.sep)
    return path
