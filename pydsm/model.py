# -*- coding: utf-8 -*-
from importlib import reload
from collections import defaultdict
import codecs
import pickle
import math
import abc
import io
import bz2

import numpy as np
import sys
import pydsm

from pydsm.utils import timer, tokenize
import pydsm.utils as utils
from pydsm.indexmatrix import IndexMatrix
import pydsm.composition as composition
import pydsm.similarity as similarity
import pydsm.weighting as weighting


def _flatten(context):
    flattened = []
    for c in context:
        if isinstance(c, list):
            for ngram in c:
                flattened.append(ngram)
        else:
            flattened.append(c)
    return flattened


def _read_documents(corpus):
    """
    If text file, treats each line as a sentence.
    If list of list, treats each list as sentence of words
    """
    if isinstance(corpus, str):
        corpus = open(corpus)

    for sentence in corpus:
        if isinstance(sentence, list):
            yield sentence
        else:
            yield list(_tokenize(sentence))

    if isinstance(corpus, io.TextIOBase):
        corpus.close()


def _tokenize(s):
    """
    Removes all URL's replacing them with 'URL'. Only keeps A-Ö 0-9.
    """
    return tokenize(s)


class DSM(metaclass=abc.ABCMeta):

    def __init__(self, matrix, corpus, window_size, vocabulary, config, **kwargs):
        if len(window_size) != 2:
            raise TypeError("Window size must be a tuple of length 2.")
        self.window_size = tuple(window_size)

        if config:
            self.config = config
        else:
            self.config = {}

        if vocabulary:
            self.vocabulary = vocabulary

        for key, val in kwargs.items():
            if val is not None:
                self.config[key] = val

        if matrix is None:
            with timer():
                print('Building collocation matrix from corpus...', end="")
                colloc_dict = self._build(self._vocabularizer(corpus))
                self._filter_threshold_words(colloc_dict)
                self.matrix = IndexMatrix(colloc_dict)
                print()
        else:
            self.matrix = matrix

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
        pickle.dump(self, bz2.open(filepath, 'wb'))

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
        return type(self)(corpus=self.corpus,
                          window_size=self.window_size,
                          matrix=matrix,
                          vocabulary=self.vocabulary,
                          config=self.config)

    def _vocabularizer(self, corpus):
        """
        Wraps the corpus object creating a generator that counts the vocabulary, 
        and yields the focus word along with left and right context.
        Lists as replacements of words are treated as one unit and iterated through (good for ngrams).
        """

        for n, sentence in enumerate(_read_documents(corpus)):
            if n % 1000 == 0:
                print(".", end=" ")
            for i, focuses in enumerate(sentence):
                if isinstance(focuses, str):
                    focuses = [focuses]
                for focus in focuses:
                    self.vocabulary[focus] += 1
                    left = i - self.window_size[0] if i - self.window_size[0] > 0 else 0
                    right = i + self.window_size[1] + 1 if i + self.window_size[1] + 1 <= len(sentence) else len(sentence)
                    #flatten lists if contains ngrams
                    context_left = _flatten(sentence[left:i])
                    context_right = _flatten(sentence[i + 1:right])
                    yield focus, context_left, context_right


    def _filter_threshold_words(self, colloc_dict):
        """
        Removes words in the colloc_dict that are too high or low.
        """

        lower_threshold = self.config.get('lower_threshold', 0)
        higher_threshold = self.config.get('higher_threshold', float("inf"))
        for word, freq in self.vocabulary.items():
            if not lower_threshold <= freq <= higher_threshold:
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


    def apply_weighting(self, weight_func=weighting.ppmi):
        """
        Apply one of the weighting functions available in pydsm.weighting.
        """

        return self._new_instance(weight_func(self.matrix))


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
    def _build(self, text):
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
                 corpus,
                 window_size,
                 matrix=None,
                 vocabulary=None,
                 lower_threshold=None,
                 higher_threshold=None,
                 config=None,
                 **kwargs):
        """
        Builds a co-occurrence matrix from text iterator. 
        If tokenized, it should come as a list of sentence lists of words.
        If not tokenized, it treats each line in corpus as a new document.
        Distributional vectors are retrievable through mat['word']
        """
        super(type(self), self).__init__(matrix,
                                         corpus,
                                         window_size, 
                                         vocabulary,
                                         config,
                                         lower_threshold=lower_threshold, 
                                         higher_threshold=higher_threshold)

    def _build(self, text):
        """
        Builds the co-occurrence matrix from text.
        Each line in text is treated as a separate document.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        for focus, context_left, context_right in text:
            for context in context_left + context_right:
                colfreqs[focus][context] += 1

        return colfreqs


class RandomIndexing(DSM):
    def __init__(self,
                 corpus,
                 window_size,
                 config=None,
                 lower_threshold=None,
                 higher_threshold=None,
                 tokenized=False,
                 dimensionality=2000,
                 num_indices=8,
                 vocabulary=None,
                 matrix=None,
                 **kwargs):
        """
        Builds a Random Indexing DSM. 
        Distributional vectors are retrievable through mat['word']
        """
        super(type(self), self).__init__(matrix,
                                         corpus,
                                         window_size, 
                                         vocabulary,
                                         config,
                                         lower_threshold=lower_threshold, 
                                         higher_threshold=higher_threshold,
                                         dimensionality=dimensionality,
                                         num_indices=num_indices)

    def _build(self, text):
        """
        Builds the co-occurrence dict from text.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        # Stores the vocabulary with frequency
        word_to_col = dict()
        for focus, context_left, context_right in text:
            for context in context_left + context_right:
                if context not in word_to_col:
                    #create index vector, and seed random state with context word
                    index_vector = set()
                    # Hash function must be between 0 and 4294967295
                    seed = hash(context) % 4294967295
                    np.random.seed(seed)
                    while len(index_vector) < self.config['num_indices']:
                        index_vector.add(np.random.random_integers(0, self.config['dimensionality'] - 1))
                        index_vector.add(-1 * np.random.random_integers(0, self.config['dimensionality'] - 1))
                    word_to_col[context] = index_vector

                # add 1 to each context. addition or subtraction is decided by the sign of the index.
                for j in word_to_col[context]:
                    colfreqs[focus][abs(j)] += math.copysign(1, j)

        return colfreqs
