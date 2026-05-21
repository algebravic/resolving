"""Simple timing"""

import time

class Timer:
    """Simple timing context"""

    def __init__(self, name):
        self._name = name
        self._start = 0.0
    def __enter__(self):
        self._start = time.time()
    def __exit__(self, *exc_info):
        elapsed = time.time() - self._start
        print(f"Elapsed time for {self._name} is {elapsed}")
