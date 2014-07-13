# -*- coding: utf-8 -*-
from importlib import reload
from collections import defaultdict
import codecs
import pickle
import math
from types import MethodType

from scipy.sparse import coo_matrix
import numpy as np
import sys

from pydsm.utils import timer, tokenize
import pydsm.utils as utils
import pydsm.similarity as similarity
import pydsm.composition as composition
import pydsm.visualization as visualization
import pydsm.weighting as weighting
from pydsm.matrix import Matrix

import_operators = [similarity, visualization, weighting, composition]


def _read_documents(filepath):
    """
    Treats each line as a document.
    """
    with codecs.open(filepath, encoding='utf-8') as f:
        for document in f:
            yield list(_tokenize(document))


def _tokenize(s):
    """
    Removes all URL's replacing them with 'URL'. Only keeps A-Ö 0-9.
    """
    return tokenize(s)


class DSM(object):
    def __init__(self, *args):
        if type(self) == DSM:
            raise NotImplementedError("Use one of the subclasses to DSM.")
        for operators in import_operators:
            # Stores the vocabulary with frequency
            self.vocabulary = defaultdict(int)
            self._import_operators(operators)


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


    def store(self, filepath):
        pickle.dump(self, open(filepath + '.dsm', 'wb'))

    @property
    def config(self):
        """
        Returns the configuration of the DSM.
        :return: Dict of configuration settings.
        """
        return self._config

    @config.setter
    def config(self, config):
        """
        Returns the configuration of the DSM.
        :return: Dict of configuration settings.
        """
        self._config = config

    def add_to_config(self, attribute, value):
            self._config[attribute].append(value)

    def _import_operators(self, operators):
        """
        The purpose of this function is to import operators of the given modules.
        Doing it this way means that you don't have to retrain a stored DSM every time new functions are added or fixed.
        Make sure the given module has a __dsm__ variable that defines which functions and attributes to be imported.
        """
        reload(operators)

        for func_name in operators.__dsm__:  # __all__ contains all functions and attributes to import
            func = getattr(operators, func_name)
            if callable(func):  # is a function, add as instance method
                method = MethodType(func, self)
                setattr(self, func_name, method)
            else:  # is an attr, add as attribute
                setattr(self, func_name, func)


    def __getitem__(self, arg):
        return self.matrix[arg]

    def __repr__(self):
        res = "{}\nVocab size: {}\n{}".format(type(self).__name__, len(self.vocabulary), self.matrix.print_matrix(3,3))
        return res

    def __str__(self):
        return self.matrix.__str__()


class CooccurrenceDSM(DSM):
    def __init__(self, corpus_path, window_size, matrix=None, config=None, vocabulary=None):
        """
        Builds a co-occurrence matrix from file. It treats each line in corpus_path as a new document.
        Distributional vectors are retrievable through mat['word']
        """
        super(type(self), self).__init__()
        if len(window_size) != 2:
            raise TypeError("Window size must be a tuple of length 2.")
        self.window_size = tuple(window_size)
        self.corpus_path = corpus_path
        if matrix is None:
            with timer():
                print('Building co-occurrence matrix from corpus...', end="")
                self.matrix = self._build(corpus_path)
                print()
        else:
            self.matrix = matrix

        if vocabulary:
            self.vocabulary = vocabulary

        if config is None:
            self._config = defaultdict(list)
            self.add_to_config('window_size', window_size)
            self.add_to_config('corpus_path', corpus_path)
        else:
            self._config = config

    def _new_instance(self, matrix, add_to_config=None):
        new_config = self._config.copy()
        if add_to_config:
            for attr, val in add_to_config.items():
                new_config[attr].append(val)

        return CooccurrenceDSM(corpus_path=self.corpus_path,
                               window_size=self.window_size,
                               matrix=matrix,
                               config=new_config,
                               vocabulary=self.vocabulary)



    def _build(self, filepath):
        """
        Builds the co-occurrence matrix from filepath.
        Each line in filepath is treated as a separate document.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        n_rows = utils.count_rows(filepath)
        bar = utils.ProgressBar(n_rows)

        for n, doc in enumerate(_read_documents(filepath)):
            bar.setAndPlot(n)
            for i, focus in enumerate(doc):
                self.vocabulary[focus] += 1
                left = i - self.window_size[0] if i - self.window_size[0] > 0 else 0
                right = i + self.window_size[1] + 1 if i + self.window_size[1] + 1 <= len(doc) else len(doc)
                for context in doc[left:i] + doc[i + 1:right]:
                    colfreqs[focus][context] += 1


        return Matrix(colfreqs)


class RandomIndexing(DSM):
    def __init__(self, corpus_path, window_size, dimensionality=2000, num_indices=8, vocabulary=None, matrix=None, config=None):
        """
        Builds a Random Indexing DSM from file. It treats each line in corpuspath as a new document.
        Currently quite slow.
        Distributional vectors are retrievable through mat['word']
        """
        super(type(self), self).__init__()
        if len(window_size) != 2:
            raise TypeError("Window size must be a tuple of length 2.")

        self.corpus_path = corpus_path
        self.window_size = tuple(window_size)
        self.dimensionality = dimensionality
        self.num_indices = num_indices
        self._config = defaultdict(list)
        if vocabulary:
            self.vocabulary = vocabulary

        if config is None:
            self.add_to_config('corpus_path', corpus_path)
            self.add_to_config('window_size', window_size)
            self.add_to_config('dimensionality', dimensionality)
            self.add_to_config('num_indices', num_indices)
        else:
            self._config = config

        if matrix is None:
            with timer():
                print('Building co-occurrence matrix from corpus...', end="")
                self.matrix = self._build(corpus_path)
                print()
        else:
            self.matrix = matrix

    def _new_instance(self, matrix, add_to_config):
        new_config = self._config.copy()
        if add_to_config:
            for attr, val in add_to_config.items():
                new_config[attr].append(val)

        return RandomIndexing(matrix=matrix,
                              corpus_path=self.corpus_path,
                              window_size=self.window_size,
                              dimensionality=self.dimensionality,
                              num_indices=self.num_indices,
                              vocabulary=self.vocabulary,
                              config=new_config)

    def _build(self, filepath):
        """
        Builds the co-occurrence matrix from filepath.
        Each line in filepath is treated as a separate document.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        # Stores the vocabulary with frequency
        word_to_col = dict()

        n_rows = utils.count_rows(filepath)
        bar = utils.ProgressBar(n_rows)

        for n, doc in enumerate(_read_documents(filepath)):
            bar.setAndPlot(n)
            for i, focus in enumerate(doc):
                self.vocabulary[focus] += 1
                left = i - self.window_size[0] if i - self.window_size[0] > 0 else 0
                right = i + self.window_size[1] + 1 if i + self.window_size[1] + 1 <= len(doc) else len(doc)

                for context in doc[left:i] + doc[i + 1:right]:
                    if context not in word_to_col:
                        #create index vector, and seed random state with context word
                        index_vector = set()
                        # Hash function may return negatives which the seed cannot handle.
                        np.random.seed(sys.maxsize + 1 + hash(context))
                        while len(index_vector) < self.num_indices:
                            index_vector.add(np.random.random_integers(0, self.dimensionality - 1))
                            index_vector.add(-1 * np.random.random_integers(0, self.dimensionality - 1))
                        word_to_col[context] = index_vector

                    # add 1 to each context. addition or subtraction is decided by the sign of the index.
                    for j in word_to_col[context]:
                        colfreqs[focus][abs(j)] += math.copysign(1, j)

        return Matrix(colfreqs)