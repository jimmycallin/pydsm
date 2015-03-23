# -*- coding: utf-8 -*-

__title__ = 'pydsm'
__version__ = '0.1'
__author__ = 'Jimmy Callin'

import pickle
import bz2
from pydsm.model import CooccurrenceDSM
from pydsm.model import RandomIndexing
from pydsm.indexmatrix import IndexMatrix


def load(filepath):
    return pickle.load(bz2.open(filepath, 'rb'))


def build(model,
          corpus,
          config=None,
          **kwargs):
    """
    Builds a distributional semantic model.
    Parameters:
        model: A semantic model class.
        Available models:
          CooccurrenceDSM
          RandomIndexing
        corpus: Either a path to file or an iterable.

    Returns: A DSM.
    """
    if config is None:
      config = {}
    config = dict(config, **kwargs)
    return model(corpus=corpus, **config)