# -*- coding: utf-8 -*-
from importlib import reload
from collections import defaultdict
import codecs
import pickle
import math
import abc

import numpy as np
import sys
import pydsm

from pydsm.utils import timer, tokenize
import pydsm.utils as utils
from pydsm.matrix import Matrix
import pydsm.composition as composition
import pydsm.similarity as similarity
import pydsm.weighting as weighting


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


class DSM(metaclass=abc.ABCMeta):

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

    def add_to_config(self, attribute, value):
            self._config[attribute].append(value)


    def compose(dsm, w1, w2, comp_func=composition.linear_additive, **kwargs):
        """
        Returns a space containing the distributional vector of a composed word pair.
        The composition type is decided by comp_func.
        """
        if isinstance(w1, str):
            w1_string = w1
            vector1 = dsm[w1]
        elif isinstance(w1, Matrix) and w1.is_vector():
            w1_string = w1.row2word[0]
            vector1 = w1

        if isinstance(w2, str):
            w2_string = w2
            vector2 = dsm[w2]
        elif isinstance(w2, Matrix) and w2.is_vector():
            w2_string = w2.row2word[0]
            vector2 = w2

        res_vector = comp_func(vector1, vector2, **kwargs)

        return res_vector


    def apply_weighting(dsm, weight_func=weighting.ppmi):
        """
        Apply one of the weighting functions available in pydsm.weighting.
        """

        return dsm._new_instance(weight_func(dsm.matrix), add_to_config={'weighting': weight_func})


    def nearest_neighbors(dsm, arg, simfunc=similarity.cos):
        vec = None

        if isinstance(arg, Matrix):
            vec = arg
        else:
            vec = dsm[arg]

        scores = []
        for row in vec:
            scores.append(simfunc(dsm.matrix, row).sort(key='sum', axis=0, ascending=False))


        res = scores[0]
        for i in scores[1:]:
            res = res.append(i, axis=1)
        return res


    @abc.abstractmethod
    def _build(self, filepath):
        """
        Builds a distributional semantic model from file. The file needs to be one document per row.
        """
        return

    @abc.abstractmethod
    def _new_instance(self, matrix, add_to_config=None):
        """
        Creates a new instance of the class containing the same configuration except for the given matrix and
        possibly additional configuration arguments. This is used for e.g. creating a new instance after applying
        a weighting function on the DSM..
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
                 corpus_path,
                 window_size,
                 matrix=None,
                 config=None,
                 vocabulary=None):
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
    def __init__(self,
                 corpus_path,
                 window_size,
                 dimensionality=2000,
                 num_indices=8,
                 vocabulary=None,
                 matrix=None,
                 config=None):
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