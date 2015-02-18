# -*- coding: utf-8 -*-
from importlib import reload
from collections import defaultdict
import codecs
import pickle
import math
import abc
import io
import bz2
import hashlib 

from itertools import chain

import numpy as np
import scipy.sparse as sp
import sys
import pydsm

from pydsm.utils import timer, tokenize
from pydsm.indexmatrix import IndexMatrix
from . import composition
from . import similarity
from . import weighting
from .cmodel import _vocabularize

class DSM(metaclass=abc.ABCMeta):

    def __init__(self, 
                 matrix=None, 
                 corpus=None, 
                 config=None, 
                 **kwargs):

        if config is None:
            self.config = {}
        
        self.config = dict(config, **kwargs)

        if matrix:
            self.matrix = matrix
        else:
            if corpus is not None:
                with timer():
                    print('Building matrix from corpus with config: {}'.format(self.config), end="")
                    colloc_dict = self.build(_vocabularize(self, corpus))
                    if isinstance(colloc_dict, dict):
                        self._filter_threshold_words(colloc_dict)
                        self.matrix = IndexMatrix(colloc_dict)
                    elif isinstance(colloc_dict, tuple):
                        self.matrix = IndexMatrix(*colloc_dict)
                    print()
            else:
              self.matrix = IndexMatrix({})

    @property
    def col2word(self):
        return self.matrix.col2word

    @property
    def row2word(self):
        return self.matrix.row2word

    @property
    def word2col(self):
        return self.matrix.word2col

    @property
    def word2row(self):
        return self.matrix.word2row


    @property
    def matrix(self):
        if not hasattr(self, '_matrix'):
            self._matrix = None
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value
    

    def store(self, filepath):
        pickle.dump(self, bz2.open(filepath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def vocabulary(self):
        """
        A corpus frequency dictionary.
        """
        if not hasattr(self, '_vocabulary') or self._vocabulary is None:
            self._vocabulary = defaultdict(int)
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, dict_like):
        self._vocabulary = defaultdict(int, dict_like)

    def _new_instance(self, matrix):
        return type(self)(matrix=matrix,
                          config=self.config)


    def _filter_threshold_words(self, colloc_dict):
        """
        Removes words in the colloc_dict that are too high or low.
        """

        lower_threshold = self.config.get('lower_threshold', 0)
        higher_threshold = self.config.get('higher_threshold', float("inf"))
        for word, freq in self.vocabulary.items():
            if not lower_threshold <= freq <= higher_threshold:
                if word in colloc_dict:
                    del colloc_dict[word]
                for key in colloc_dict.keys():
                    if word in colloc_dict[key]:
                        del colloc_dict[key][word]

    def compose(self, w1, w2, comp_func=composition.linear_additive, **kwargs):
        """
        Returns a space containing the distributional vector of a composed word pair.
        The composition type is decided by comp_func.
        """
        if isinstance(w1, str):
            w1_string = w1
            vector1 = self[w1]
        elif isinstance(w1, IndexMatrix) and w1.is_vector():
            w1_string = w1.row2word[0]
            vector1 = w1

        if isinstance(w2, str):
            w2_string = w2
            vector2 = self[w2]
        elif isinstance(w2, IndexMatrix) and w2.is_vector():
            w2_string = w2.row2word[0]
            vector2 = w2

        res_vector = comp_func(vector1, vector2, **kwargs)

        return res_vector


    def apply_weighting(self, weight_func=weighting.ppmi, *args):
        """
        Apply one of the weighting functions available in pydsm.weighting.
        """

        return self._new_instance(weight_func(self.matrix), *args)


    def nearest_neighbors(self, arg, sim_func=similarity.cos):
        vec = None

        if isinstance(arg, IndexMatrix):
            vec = arg
        else:
            vec = self[arg]

        scores = []
        for row in vec:
            scores.append(sim_func(self.matrix, row).sort(key='sum', axis=0, ascending=False))


        res = scores[0]
        for i in scores[1:]:
            res = res.append(i, axis=1)
        return res


    @abc.abstractmethod
    def build(self, text):
        """
        Builds a distributional semantic model from file. The file needs to be one document per row.
        """
        return

    def __getitem__(self, arg):
        return self.matrix[arg]

    def __repr__(self):
        res = "{}\nVocab size: {}\n{}".format(type(self).__name__, len(self.vocabulary), self.matrix.print_matrix(3,3))
        return res

    def __str__(self):
        return self.matrix.__str__()


class CooccurrenceDSM(DSM):
    def __init__(self,
                 matrix=None,
                 corpus=None,
                 config=None,
                 **kwargs):
        """
        Builds a co-occurrence matrix from text iterator. 
        Parameters:
        window_size: 2-tuple of size of the context
        matrix: Instantiate DSM with already created matrix.
        lower_threshold: Minimum frequency of word for it to be included.
        higher_threshold: Maximum frequency of word for it to be included.
        ordered: Differentates between context words in different positions. 
        directed: Differentiates between left and right context words.
        """
        if config is None:
            config = {}
        config = dict(config, **kwargs)

        super(type(self), self).__init__(matrix,
                                         corpus,
                                         config)

    def build(self, text):
        """
        Builds the co-occurrence matrix from text.
        Each line in text is treated as a separate document.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        for focus, contexts in text:
            for context in contexts:
                colfreqs[focus][context] += 1

        return colfreqs


class RandomIndexing(DSM):
    def __init__(self,
                 matrix=None,
                 corpus=None,
                 config=None,
                 **kwargs):
        """
        Builds a Random Indexing DSM from text-iterator. 
        Parameters:
        window_size: 2-tuple of size of the context
        matrix: Instantiate DSM with already created matrix.
        vocabulary: When building, the DSM also creates a frequency dictionary. 
                    If you include a matrix, you also might want to include a frequency dictionary
        lower_threshold: Minimum frequency of word for it to be included.
        higher_threshold: Maximum frequency of word for it to be included.
        ordered: Differentates between context words in different positions. 
        directed: Differentiates between left and right context words.
        dimensionality: Number of columns in matrix.
        num_indices: Number of positive indices, as well as number of negative indices.
        """
        if config is None:
            config = {}
        config = dict(config, **kwargs)
        super().__init__(matrix=matrix,
                         corpus=corpus,
                         config=config)


    def build(self, text):
        """
        Builds the co-occurrence dict from text.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        # Stores the vocabulary with frequency
        index_vectors = dict()
        for focus, contexts in text:
            for context in contexts:
                if context not in index_vectors:
                    #create index vector, and seed random state with context word
                    index_vector = set()
                    # Hash function must be between 0 and 4294967295
                    hsh = hashlib.md5()
                    hsh.update(context.encode())
                    seed = int(hsh.hexdigest(), 16) % 4294967295 # highest number allowed by seed
                    np.random.seed(seed)
                    while len(index_vector) < self.config['num_indices']:
                        index_vector.add(np.random.random_integers(0, self.config['dimensionality'] - 1))
                        index_vector.add(-1 * np.random.random_integers(0, self.config['dimensionality'] - 1))
                    index_vectors[context] = index_vector

                # add 1 to each context. addition or subtraction is decided by the sign of the index.
                for j in index_vectors[context]:
                    colfreqs[focus][abs(j)] += math.copysign(1, j)

        # Make sure all indices are created for at least one vector
        for w, vector in colfreqs.items():
            for i in range(self.config['dimensionality']):
                if i not in vector:
                    vector[i] = 0
            break

        return colfreqs