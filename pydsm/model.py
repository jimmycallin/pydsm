# -*- coding: utf-8 -*-
from collections import defaultdict
import pickle
import math
import abc
import bz2
import hashlib

import numpy as np

from .utils import timeit
from .utils import to_dict_tree
from .indexmatrix import IndexMatrix
from . import composition
from . import similarity
from . import weighting
from . import evaluation
from . import visualization
from .cmodel import _vocabularize


class DSM(metaclass=abc.ABCMeta):

    def __init__(self,
                 matrix=None,
                 corpus=None,
                 config=None,
                 vocabulary=None,
                 **kwargs):

        if config is None:
            self.config = {}

        self.config = dict(config, **kwargs)

        if matrix:
            self.matrix = matrix
        else:
            if corpus is not None:
                print('Building matrix from corpus with config: {}'.format(self.config), end="")
                colloc_dict = self.build(_vocabularize(self, corpus))
                if isinstance(colloc_dict, dict):
                    if 'higher_threshold' in self.config:
                        self._filter_high_threshold_words(colloc_dict)
                    self.matrix = IndexMatrix(colloc_dict)
                elif isinstance(colloc_dict, tuple):
                    self.matrix = IndexMatrix(*colloc_dict)
                elif isinstance(colloc_dict, IndexMatrix):
                    self.matrix = colloc_dict
                else:
                    raise ValueError(
                        "The model needs to build a dict of dict, a matrix-row2word-col2word tuple, or an IndexMatrix")
                print()

            else:
                self.matrix = IndexMatrix({})

        if vocabulary:
            self.vocabulary = vocabulary

    @property
    def col2word(self):
        """
        A list of words, where the index of a word correlates to the index of the column.
        :return: List of column indices.
        """
        return self.matrix.col2word

    @property
    def row2word(self):
        """
        A list of words, where the index of a word correlates to the index of the row.
        :return: List of row indices.
        """
        return self.matrix.row2word

    @property
    def word2col(self):
        """
        A dict of words along the col axis, containing the index of which col that correlates to which word.
        This is only generated when necessary.
        :return: Dict of word -> col index.
        """
        return self.matrix.word2col

    @property
    def word2row(self):
        """
        A dict of words along the row axis, containing the index of which row that correlates to which word.
        This is only generated when necessary.
        :return: Dict of word -> row index.
        """
        return self.matrix.word2row

    @property
    def matrix(self):
        """
        :return: A DSM matrix.
        """
        if not hasattr(self, '_matrix'):
            self._matrix = None
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    def store(self, filepath):
        """
        Stores a BZ2 compressed binary of self using pickle.
        Can be loaded into memory using pydsm.load
        """
        pickle.dump(self, bz2.open(filepath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def vocabulary(self):
        """
        :return: Word frequency dictionary.
        """
        if not hasattr(self, '_vocabulary') or self._vocabulary is None:
            self._vocabulary = defaultdict(int)
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, dict_like):
        self._vocabulary = defaultdict(int, dict_like)

    def _new_instance(self, matrix):
        return type(self)(matrix=matrix,
                          config=self.config,
                          vocabulary=self.vocabulary)

    def _filter_high_threshold_words(self, colloc_dict):
        """
        Removes words in the colloc_dict that have too high of a frequency.
        """
        higher_threshold = self.config.get('higher_threshold', None)
        if higher_threshold:
            for word, freq in self.vocabulary.items():
                if freq > higher_threshold:
                    if word in colloc_dict:
                        del colloc_dict[word]
                    for key in colloc_dict.keys():
                        if word in colloc_dict[key]:
                            del colloc_dict[key][word]

    def compose(self, w1, w2, comp_func=composition.linear_additive, **kwargs):
        """
        Returns a space containing the distributional vector of a composed word pair.
        The composition type is decided by comp_func.
        :param w1: First word or IndexMatrix vector to compose.
        :param w2: Second word or IndexMatrix vector to compose.
        :return: Composed vector.
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

    @timeit
    def apply_weighting(self, weight_func=weighting.ppmi, **kwargs):
        """
        Apply one of the weighting functions available in pydsm.weighting.
        :param weight_func: The weighting function to apply. First argument has to be of type IndexMatrix.
        Additional set parameters are sent to the weighting function.
        """

        return self._new_instance(weight_func(self.matrix, **kwargs))

    @timeit
    def evaluate(self, evaluation_test=evaluation.simlex, sim_func=similarity.cos, **kwargs):
        """
        Evaluate the model given an evaluation function.
        The evaluation functions are available in pydsm.evaluation.
        Additional set parameters are sent to the evaluation function.
        """

        return evaluation_test(self.matrix, sim_func=sim_func, **kwargs)

    def visualize(self, vis_func=visualization.heatmap, **kwargs):
        """
        Plot a DSM given a visualization function.
        The visualization functions are available in pydsm.visualization
        :param vis_func: Visualization function. First argument has to be of type IndexMatrix.
        Additional set parameters are sent to the visualization function.
        """

        return vis_func(self.matrix, **kwargs)

    @timeit
    def nearest_neighbors(self, arg, sim_func=similarity.cos):
        """
        Find the nearest neighbors given arg.
        :param arg: Either a string or an IndexMatrix.
                    If index matrix, return nearest neighbors to all row vectors of the matrix.
        :param sim_func: The similarity function to use for proximity calculation.
        """

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

    def relative_neighborhood(self, w, k, format='dict'):
        """
        Builds a relative neighborhood graph from word `w` using the `k` nearest neighbors of `w`.
        See [1] for details.

        [1] Sahlgren et. al. (2015) "Navigating the Semantic Horizon using Relative Neighborhood Graphs"

        :param w: Root word.
        :param k: number of nearest neighbors.
        :param format: If 'dict', a json serializable dict object.
                       If 'networkx', a networkx.DiGraph object.
        :return: A relative neighborhood graph of format according to `format`.
        """
        import networkx as nx
        # K nearest neighbors of w
        nns = self.nearest_neighbors(w)[:k]
        # KNN vectors
        vectors = self.matrix[nns.row2word]
        # Distance matrix
        sims = similarity.cos(vectors, vectors)
        # First column is distance to w,
        # by removing this value from all matrix values,
        # we can look for only positive values
        # and use largest of these as parent word
        # only look in lower triangular matrix
        diffs = (sims - sims[:, 0]).triangular_lower(k=-1)
        # filaments is implemented as directed graph
        filaments = nx.DiGraph()
        filaments.add_nodes_from(nns.row2word)
        root_word = nns.row2word[0]
        for row, word in zip(diffs, diffs.row2word):
            if word == root_word:
                continue
            index = row.matrix.data.argmax()
            max_val = row.matrix.data[index]
            if max_val > 0:
                col = row.matrix.indices[index]
                max_word = row.col2word[col]
                filaments.add_edge(max_word, word, {'score': sims[word, max_word]})
            else:
                filaments.add_edge(root_word, word, {'score': sims[word, root_word]})

        if format == 'networkx':
            return filaments
        elif format == 'dict':
            return to_dict_tree(filaments, root_word)
        else:
            raise NotImplementedError("Format {} is not supported".format(format))

    @abc.abstractmethod
    def build(self, text):
        """
        Builds a distributional semantic model from file. All models need to implement this.
        """
        raise NotImplementedError

    def __getitem__(self, arg):
        """
        Returns an arg vector of the matrix.
        """
        return self.matrix[arg]

    def __repr__(self):
        res = "{}\nVocab size: {}\n{}".format(type(self).__name__, len(self.vocabulary), self.matrix.print_matrix(3, 3))
        return res

    def __str__(self):
        return self.matrix.__str__()


class CooccurrenceDSM(DSM):

    def __init__(self,
                 matrix=None,
                 corpus=None,
                 config=None,
                 vocabulary=None,
                 window_size=(2, 2),
                 **kwargs):
        """
        Builds a co-occurrence matrix from text iterator.
        While iterating through the text, it counts coocurrence between the focus word and its neighboring words within window_size.

        :param matrix: Instantiate DSM with already created matrix.
        :param corpus: File path string or iterable to read.
        :param config: Additional configuration options.
                       Obligatory:
                           window_size: 2-tuple of size of the context
                       Optional:
                           lower_threshold: Minimum frequency of word for it to be included (default 0).
                           higher_threshold: Maximum frequency of word for it to be included (default infinite).
                           ordered: Differentates between context words in different positions (default False).
                           directed: Differentiates between left and right context words (default False).
        """
        if config is None:
            config = {}
        config = dict(config, window_size=window_size, **kwargs)

        super(type(self), self).__init__(matrix,
                                         corpus,
                                         config,
                                         vocabulary)

    @timeit
    def build(self, text):
        """
        Builds the cooccurrence matrix from text.
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
                 window_size=(2, 2),
                 vocabulary=None,
                 dimensionality=2000,
                 num_indices=8,
                 **kwargs):
        """
        Builds a Random Indexing DSM from text-iterator [1].

        :param matrix: Instantiate DSM with already created matrix.
        :param corpus: File path string or iterable to read.
        :param config: Additional configuration options.
                       Obligatory:
                           window_size: 2-tuple of size of the context.
                           dimensionality: Number of columns in matrix (default 2000).
                           num_indices: Number of positive indices, as well as number of negative indices (default 8).
                       Optional:
                           lower_threshold: Minimum frequency of word for it to be included (default 0).
                           higher_threshold: Maximum frequency of word for it to be included (default infinite).
                           ordered: Differentates between context words in different positions (default False).
                           directed: Differentiates between left and right context words (default False).

        [1] Sahlgren, Magnus. "An introduction to random indexing." (2005).
        """
        if config is None:
            config = {'dimensionality': dimensionality,
                      'num_indices': num_indices,
                      'window_size': window_size}
        else:
            if 'dimensionality' not in config:
                config['dimensionality'] = dimensionality
            if 'num_indices' not in config:
                config['num_indices'] = num_indices
            if 'window_size' not in config:
                config['window_size'] = window_size

        config = dict(config, **kwargs)

        super().__init__(matrix=matrix,
                         corpus=corpus,
                         config=config)

    @timeit
    def build(self, text):
        """
        Builds the co-occurrence dict from text.
        The columns are encoded from 0 to dimensionality-1.
        """
        # Collect word collocation frequencies in dict of dict
        colfreqs = defaultdict(lambda: defaultdict(int))

        # Stores the vocabulary with frequency
        index_vectors = dict()
        for focus, contexts in text:
            for context in contexts:
                if context not in index_vectors:
                    # create index vector, and seed random state with context word
                    index_vector = set()
                    # Hash function must be between 0 and 4294967295
                    hsh = hashlib.md5()
                    hsh.update(context.encode())
                    seed = int(hsh.hexdigest(), 16) % 4294967295  # highest number allowed by seed
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
