import os

from . import __file__


def _get_resource(fname):
    base = os.path.dirname(os.path.realpath(__file__))

    return os.path.join(base, fname)
