import sys
from contextlib import contextmanager

@contextmanager
def temporary_sys_path(path):
    original_sys_path = sys.path.copy()
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path = original_sys_path