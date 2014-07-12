# -*- coding: utf-8 -*-
from importlib import reload
from collections import defaultdict
import codecs
import pickle
import math
from types import MethodType

from scipy.sparse import coo_matrix
import numpy as np

from pydsm.utils import timer, tokenize
import pydsm.utils as utils
import pydsm.similarity as similarity
import pydsm.composition as composition
import pydsm.visualization as visualization
import pydsm.weighting as weighting
from pydsm.matrix import Matrix

import_operators = [similarity, visualization, weighting, composition]


class DSM(object):
    def __init__(self, *args):
        if type(self) == DSM:
            raise NotImplementedError("Use one of the subclasses to DSM.")
        for operators in import_operators:
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


    def store(self, filepath, file_type='binary'):
        if file_type == 'binary':
            pickle.dump(self, open(filepath + '.bcm', 'wb'))

    @property
    def config(self):
        """
        Returns the configuration of the DSM.
        :return: Dict of configuration settings.
        """
        return self._config

    @config.setter
    def config(self, configs):
        self._config = configs

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


    @staticmethod
    def _tokenize(s):
        """
        Removes all URL's replacing them with 'URL'. Only keeps A-Ö 0-9.
        """
        return tokenize(s)

    def _read_documents(self, filepath):
        """
        Treats each line as a document.
        """
        with codecs.open(filepath, encoding='utf-8') as f:
            for document in f:
                yield list(self._tokenize(document))
                # TODO: Add hashed value of documents.

    @staticmethod
    def _to_space(mat, word_to_row, word_to_col=None, columns=None):
        if columns is None:
            columns = sorted(word_to_col.keys(), key=lambda word: word_to_col[word])
        else:
            columns = columns

        index = sorted(word_to_row.keys(), key=lambda word: word_to_row[word])

        matrix = Matrix(mat, col2word=columns, row2word=index)

        return matrix

    def _to_coo(self, colfreqs):
        pass

    def __getitem__(self, arg):
        return self.matrix[arg]

    def __repr__(self):
        return self.matrix.__repr__()

    def __str__(self):
        return self.matrix.__str__()


class CooccurrenceDSM(DSM):
    def __init__(self, corpus_path, window_size, matrix=None, config=None):
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

        self.config = config if config else {}
        self.config.update({'window_size': window_size,
                            'corpus_path': corpus_path})

    def _new_instance(self, matrix, add_to_config=None):
        new_config = self.config.copy()
        if add_to_config:
            for k, v in add_to_config.items():
                if k not in new_config:
                    new_config[k] = v
                else:
                    new_config[k] = list(new_config[k])
                    new_config[k].append(v)

        return CooccurrenceDSM(corpus_path=self.corpus_path,
                               window_size=self.window_size,
                               matrix=matrix,
                               config=new_config)



    def __str__(self):
        return self.matrix.__str__()

    def __repr__(self):
        return self.__str__()

    def _build(self, filepath):
        """
        Builds the co-occurrence matrix from filepath.
        Each line in filepath is treated as a separate document.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        # Stores the vocabulary with frequency
        self.vocabulary = defaultdict(int)

        n_rows = utils.count_rows(filepath)
        bar = utils.ProgressBar(n_rows)

        for n, doc in enumerate(self._read_documents(filepath)):
            bar.setAndPlot(n)
            for i, focus in enumerate(doc):
                self.vocabulary[focus] += 1
                left = i - self.window_size[0] if i - self.window_size[0] > 0 else 0
                right = i + self.window_size[1] + 1 if i + self.window_size[1] + 1 <= len(doc) else len(doc)
                for context in doc[left:i] + doc[i + 1:right]:
                    colfreqs[focus][context] += 1

        # Giving indices to words
        word_to_col = {word: index for index, word in enumerate(self.vocabulary.keys())}
        word_to_row = {word: index for index, word in enumerate(self.vocabulary.keys())}

        # Convert to coo matrix
        rows = []
        cols = []
        data = []
        for row in colfreqs.keys():
            for col in colfreqs[row].keys():
                rows.append(word_to_row[row])
                cols.append(word_to_col[col])
                data.append(colfreqs[row][col])

        mat = coo_matrix((data, (rows, cols)), shape=(len(word_to_row), len(word_to_col)))

        self.matrix = self._to_space(mat, word_to_row, word_to_col=word_to_col)
        return self.matrix


class RandomIndexing(DSM):
    def __init__(self, corpus_path, window_size, dimensionality=2000, num_indices=8, matrix=None):
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
        if matrix is None:
            with timer():
                print('Building co-occurrence matrix from corpus...', end="")
                self.matrix = self._build(corpus_path)
                print()
        else:
            self.matrix = matrix

    def _new_instance(self, matrix, add_to_config):
        return RandomIndexing(matrix=matrix,
                              corpus_path=self.corpus_path,
                              window_size=self.window_size,
                              dimensionality=self.dimensionality,
                              num_indices=self.num_indices)

    def _build(self, filepath):
        """
        Builds the co-occurrence matrix from filepath.
        Each line in filepath is treated as a separate document.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        # Stores the vocabulary with frequency
        self.vocabulary = defaultdict(int)
        word_to_col = dict()

        n_rows = utils.count_rows(filepath)
        bar = utils.ProgressBar(n_rows)

        for n, doc in enumerate(self._read_documents(filepath)):
            bar.setAndPlot(n)
            if n % 10000 == 0:
                print(".", end="")
            for i, focus in enumerate(doc):
                self.vocabulary[focus] += 1
                left = i - self.window_size[0] if i - self.window_size[0] > 0 else 0
                right = i + self.window_size[1] + 1 if i + self.window_size[1] + 1 <= len(doc) else len(doc)

                for context in doc[left:i] + doc[i + 1:right]:
                    if context not in word_to_col:
                        #create index vector
                        index_vector = set()
                        while len(index_vector) < self.num_indices:
                            index_vector.add(np.random.random_integers(0, self.dimensionality - 1))
                            index_vector.add(-1 * np.random.random_integers(0, self.dimensionality - 1))
                        word_to_col[context] = index_vector

                    # add 1 to each context. addition or subtraction is decided by the sign of the index.
                    for j in word_to_col[context]:
                        colfreqs[focus][abs(j)] += math.copysign(1, j)

        # Giving indices to words
        word_to_row = {word: index for index, word in enumerate(self.vocabulary.keys())}
        # Saving this on DSM level for now. TODO: Figure out where to put it later.
        self.word_to_col = word_to_col

        # Store as sparse coo matrix
        rows = []
        cols = []
        data = []
        for row in colfreqs.keys():
            for col in colfreqs[row].keys():
                rows.append(word_to_row[row])
                cols.append(col)
                data.append(colfreqs[row][col])

        mat = coo_matrix((data, (rows, cols)), shape=(len(word_to_row), self.dimensionality))
        return self._to_space(mat, word_to_row=word_to_row, columns=list(range(self.dimensionality)))


def load_matrix(filepath):
    return pickle.load(open(filepath, 'rb'))
