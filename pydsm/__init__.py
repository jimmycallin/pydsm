# -*- coding: utf-8 -*-

__title__ = 'pydsm'
__version__ = '0.1'
__author__ = 'Jimmy Callin'
__copyright__ = 'Copyright 2014 Jimmy Callin'

from sklearn.externals import joblib as _joblib

from pydsm.models import CooccurrenceDSM
from pydsm.models import RandomIndexing
from pydsm.matrix import Matrix

_mem = _joblib.Memory(cachedir='/tmp/pydsm')


#def load_cached(cached_id):
#    pass


#def list_cached():
#    pass


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
        window_size: Tuple of left and right window size, e.g.: (2,2)
        corpus: Path to corpus file
        language: Language of model
        store: Where to store cache files
        lemmatize: Whether or not to lemmatize the corpus
        min_ratio: Minimum word rate to appear in DSM. Cannot be set at the same time as min_freq
        max_ratio: Minimum word rate to appear in DSM. Cannot be set at the same time as max_freq
        min_freq: Minimum word frequency to appear in DSM
        max_freq: Maximum word frequency to appear in DSM
    Available models are:
        Cooccurrence Matrix: 'cooc'
        Random Indexing: 'ri'
    Returns: A DSM.
    """
    return model(corpus, window_size)


#def clear_cache():
#    _mem.clear(warn=True)