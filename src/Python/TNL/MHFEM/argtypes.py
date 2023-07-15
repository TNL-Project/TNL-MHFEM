#! /usr/bin/env python3

import argparse
import os

# path to existing directory
def argtype_existing_file(string):
    string = os.path.abspath(os.path.expanduser(string))
    if not os.path.isfile(string):
        raise argparse.ArgumentTypeError("file '%s' does not exist" % string)
    return string

def argtype_greater_than_one(string):
    try:
        num = float(string)
    except ValueError:
        raise argparse.ArgumentTypeError("cannot convert '{}' to a number".format(string))
    if num <= 1:
        raise argparse.ArgumentTypeError("expected a number greater than 1")
    return num

def argtype_positive_int(string):
    try:
        num = int(string)
    except ValueError:
        raise argparse.ArgumentTypeError("cannot convert '{}' to an integer".format(string))
    if num <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return num
