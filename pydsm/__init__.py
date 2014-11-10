# -*- coding: utf-8 -*-

__title__ = 'pydsm'
__version__ = '0.1'
__author__ = 'Jimmy Callin'
__copyright__ = 'Copyright 2014 Jimmy Callin'

import pickle
from pydsm.model import CooccurrenceDSM
from pydsm.model import RandomIndexing
from pydsm.indexmatrix import IndexMatrix


def load(filepath):
    return pickle.load(open(filepath, 'rb'))

def build(model,
          window_size,
          corpus,
          language,
          lemmatize=False,
          min_ratio=2.0e-10,
          max_ratio=0.9,
          min_freq=None,
          max_freq=None):
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
    return model(corpus, window_size)


#def clear_cache():
#    _mem.clear(warn=True)