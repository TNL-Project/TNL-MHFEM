#! /usr/bin/env python3

import contextlib
import sys

@contextlib.contextmanager
def capture():
    oldout, olderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = StringIO(), StringIO()
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = oldout, olderr

from .fd_redirector import FDRedirector

@contextlib.contextmanager
def capture_fd():
    stdout = FDRedirector(1)
    stderr = FDRedirector(2)
    stdout.start()
    stderr.start()
    # guard the yield with try-finally, otherwise the descriptors would not
    # be closed if the body of the context manager raises an exception
    try:
        yield stdout, stderr
    finally:
        stdout.stop()
        stderr.stop()
