# -*- coding: utf-8 -*-

__title__ = 'pydsm'
__version__ = '0.1'
__author__ = 'Jimmy Callin'
__copyright__ = 'Copyright 2014 Jimmy Callin'

import pickle
import bz2
from pydsm.model import CooccurrenceDSM
from pydsm.model import RandomIndexing
from pydsm.indexmatrix import IndexMatrix


def load(filepath):
    return pickle.load(bz2.open(filepath, 'rb'))


def build(model,
          window_size,
          corpus,
          lower_threshold=None,
          higher_threshold=None,
          language=None):
    """
    Builds a distributional semantic model given a set of parameters.
    Parameters:
        model: CooccurrenceDSM, or RandomIndexing.
        window_size: Tuple of left and right window size, e.g.: (2,2)
        corpus: Path to corpus file
        language: Language of model
        store: Where to store cache files
        lemmatize: Whether or not to lemmatize the corpus
        min_ratio: Minimum word rate to appear in DSM. Cannot be set at the same time as min_freq
        max_ratio: Minimum word rate to appear in DSM. Cannot be set at the same time as max_freq
        min_freq: Minimum word frequency to appear in DSM
        max_freq: Maximum word frequency to appear in DSM
    Returns: A DSM.
    """
    return model(corpus, window_size, lower_threshold=lower_threshold, higher_threshold=higher_threshold)


# def clear_cache():
#    _mem.clear(warn=True)
