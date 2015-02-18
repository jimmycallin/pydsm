# -*- coding: utf-8 -*-
from collections import defaultdict
from numbers import Number, Integral

import scipy.sparse as sp
import scipy.sparse.linalg
import numpy as np
from tabulate import tabulate

from . import cindexmatrix._dict2matrix as _dict2matrix

class IndexMatrix(object):
    """
    This is a wrapper around the scipy.sparse.spmatrix, also containing an index for the rows and columns.
    The reason for building this wrapper rather than inherit the spmatrix are several:
     1. Because of the architecture of the spmatrix, it is difficult to inherit from them.
     2. It is possible to switch between different spmatrix formats this way. Some functions might perform better as
         coo matrix, others as csr matrix.
     3. If it becomes apparent that a Dense Matrix equivalent is required in the future, it is easy to create a child
         class which implements all operators and as such has access to all other DSM methods.
     4. Probably more reasons.
    """

    def __init__(self, matrix, row2word=None, col2word=None):
        """
        Initialize a Matrix instance.
        :param matrix: A matrix of dict (row) of dict (column) of values, or of type scipy.sparse.spmatrix.
                       If matrix is not dict of dict, row2word and col2word must be set.
        :param row2word: List of words, corresponding to each row of the matrix.
        :param col2word: List of words, corresponding to each column of the matrix.
        :return:
        """
        if isinstance(matrix, dict):
            matrix, row2word, col2word = _dict2matrix(matrix)

        # Convert to sparse matrix
        if not isinstance(matrix, sp.spmatrix):
            self.matrix = sp.csr_matrix(matrix)
        else:
            self.matrix = matrix.tocsr()

        if not isinstance(row2word, list) or not isinstance(col2word, list):
            raise TypeError("Row2word and col2word needs to be of type list, not {} and {}".format(type(row2word), type(col2word)))

        if len(row2word) != matrix.shape[0]:
            raise ValueError("Matrix row is of length {}, but row2word is of length {}"\
                             .format(matrix.shape[0], len(row2word)))

        if len(col2word) != matrix.shape[1]:
            raise ValueError("Matrix column is of length {}, but col2word is of length {}"\
                                 .format(matrix.shape[1], len(col2word)))



        # Check type. Convert to double if not bool
        if not self.matrix.dtype in (np.double, np.bool):
            self.matrix = self.matrix.astype(np.double)

        self._row2word = row2word
        self._col2word = col2word

        # Only render in property when necessary
        self._word2row = None
        self._word2col = None

        self._display_max_rows = 10
        self._display_max_cols = 7

    def apply(self, func):
        """
        Apply function func element-wise on numpy.ndarray of non-zero values.
        Using numpy or scipy functions are recommended.

        Example:

        sparsemat = scipy.sparse.coo_matrix(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        mat = Matrix(sparsemat, ['one', 'two', 'three'], ['four', 'five', 'six'])
        mat.apply(np.log)
             hej                 bra                 då
        hej  0.0                 0.6931471805599453  1.0986122886681098
        bra  1.3862943611198906  1.6094379124341003  1.791759469228055
        då   1.9459101490553132  2.0794415416798357  2.1972245773362196

        :param func: Function to apply.
        :return: Matrix instance with applied function
        """
        mat = self.matrix.tocoo()
        mat.data = func(mat.data)
        return self._new_instance(mat)

    def sort(self, key='sum', axis=0, ascending=True):
        """
        Applies key function on either rows or columns, and sorts the matrix based on the resulting vector.
        :param key: Either 'sum', 'norm', Matrix vector, or a function.
                    As a function it has to have 'axis' as a parameter.
        :param axis: 0 for sorting columns, 1 for sorting rows.
        :return: Sorted Matrix instance.
        """
        if key == 'sum':
            agg = self.sum(int(not axis))
        elif key == 'norm':
            agg = self.norm(int(not axis))
        elif isinstance(key, IndexMatrix) and key.is_vector():
            agg = key
        elif callable(key):
            agg = key(self, axis=int(not axis))
        else:
            raise TypeError("Key needs to be either one of the functions, or a suitable function.")

        if ascending:
            ascending = 1
        else:
            ascending = -1

        sorted_indices = np.argsort(agg.matrix.todense().getA().flatten()).tolist()[::ascending]

        res = None
        if axis == 0:
            res = self[sorted_indices]
        elif axis == 1:
            res = self[:, sorted_indices]

        return res

    @property
    def row_col_data(self):
        """
        Get the COO representation of the matrix, that is, a tuple containing the row indices, column indices,
        and values, such that row[i], col[i], data[i] equals the value data[i] at index mat[row[i],col[i]].
        This is handy when dealing with matplotlib.
        :return: Tuple containing arrays (col, row, data)
        """
        mat = self.matrix.tocoo()
        return mat.row, mat.col, mat.data

    def delete(self, arg, axis):
        """
        Deletes arg from given axis.
        :param arg: To be deleted. Can be string, integers, list of strings or integers, etc.
        :param axis: Rows (0) or cols (1)
        :return: Matrix instance with rows deleted.
        """
        arg = self._axis2indices(arg, axis)
        mask = np.ones(self.shape[axis], dtype=bool)
        mask[arg] = False
        if axis == 0:
            return self[mask, :]
        elif axis == 1:
            return self[:, mask]
        else:
            raise ValueError("Axis can only be 0 (rows) or 1 (cols)")

    def synchronize_word_order(self, matrix, axis=None):
        """
        Sorts the column and row order of self so that it matches the given matrix.
        The indices for both matrices along the axis to be sorted must not contain unique words.
        :param matrix: Matrix whose word order to use as template.
        :param axis: 0 to synchronise rows, 1 for columns, None for both.
        :return: Sorted matrix
        """
        this_col2word = self.col2word
        this_row2word = self.row2word
        that_col2word = matrix.col2word
        that_row2word = matrix.row2word

        if axis != 0:
            if len(set.difference(set(this_col2word), set(that_col2word))) != 0:
                raise ValueError("At least one of the column indices has a unique value.")

        if axis != 1:
            if len(set.difference(set(this_row2word), set(that_row2word))) != 0:
                raise ValueError("At least one of the row indices has a unique value.")

        if axis == 0:
            return self[that_row2word, :]
        elif axis == 1:
            return self[:, that_col2word]
        elif axis is None:
            return self[that_row2word, that_col2word]
        else:
            raise ValueError("Axis must be 0, 1 or None")

    def append(self, matrix, axis):
        """
        Appends a Matrix instance either row-wise (0) or column-wise (1).
        When appended e.g. row-wise, the matrices must share the same column word index, and
        the combined row word index cannot contain duplicates.

        :param matrix: Matrix to be appended.
        :param axis: 0 if appending row-wise, 1 if column-wise.
        :return: A combined matrix.
        """

        axis_to_be_synced = int(not axis)
        mat = matrix.synchronize_word_order(self, axis_to_be_synced)

        if axis == 0:
            if len(set.intersection(set(self.row2word), set(matrix.row2word))) != 0:
                raise ValueError("The row to append doesn't only contain unique values.")

            return self._new_instance(sp.vstack([self.matrix, matrix.matrix], format='csr'),
                                      row2word=self.row2word + matrix.row2word)
        elif axis == 1:
            if len(set.intersection(set(self.col2word), set(matrix.col2word))) != 0:
                raise ValueError("The column to append doesn't only contain unique values.")

            return self._new_instance(sp.hstack([self.matrix, mat.matrix], format='csr'),
                                      col2word=self.col2word + matrix.col2word)
        else:
            raise ValueError("Axis must be 0 or 1.")

    def is_vector(self):
        """
        Checks to see whether its row is of length one, and will as such be considered to be a vector.
        :return: True if vector, False otherwise.
        """
        if self.shape[0] == 1:
            return True
        else:
            return False

    def std(self, axis):
        """
        Returns the uncorrected sample standard deviation along the given axis.
        :param axis: The axis to calculate standard deviation on.
        :return:
        """
        if axis not in (0,1):
            raise ValueError('Axis must be 0 or 1.')

        other_axis = axis
        length = self.shape[other_axis]
        sub_mean = self - self.mean(other_axis)
        return (sub_mean.multiply(sub_mean).sum(other_axis)/length).sqrt()

    def sum(self, axis):
        """
        Sums up all values along the given axis.
        :param axis: 0 for summing along the rows, 1 for summing along the cols.
        :return:
        """
        if axis not in (0, 1):
            raise ValueError("Axis must be 0 or 1")

        summed = self.matrix.sum(axis=axis)
        if axis == 0:
            return self._new_instance(summed, row2word=[''])
        elif axis == 1:
            return self._new_instance(summed, col2word=[''])
        else:
            raise ValueError("Axis may only be 0 (rows) or 1 (cols)")

    def dot(self, matrix):
        """
        Matrix multiplication.
        When Python 3.5 arrives, @ operator will be added for this.
        :param matrix: Right-side matrix to perform dot product on.
        :return: Resulted matrix.
        """
        if isinstance(matrix, IndexMatrix):
            return self._new_instance(self.matrix.dot(matrix.matrix), col2word=matrix.col2word, row2word=self.row2word)
        else:
            raise TypeError("Can only operate on Matrix instances")

    @property
    def shape(self):
        """
        The shape of the matrix.
        :return: A tuple of length 2. Index 0 represents number of rows, index 1 number of columns.
        """
        return self.matrix.shape

    def log(self):
        """
        Computes ln(x) element-wise.
        :return: Logged matrix instance.
        """
        mat = self.matrix.tocoo()
        mat.data = np.log(mat.data)
        return self._new_instance(mat)

    def plog(self):
        """
        Computes ln(x) element-wise. Where x < 1, set to 1 for avoiding log values below 0.
        :return: Logged matrix instance.
        """
        mat = self.matrix.tocoo()
        mat.data[mat.data < 1] = 1
        mat.data = np.log(mat.data)
        return self._new_instance(mat)

    def log1p(self):
        """
        Computes ln(1+x) element-wise.
        :return: Logged matrix instance.
        """
        return self._new_instance(self.matrix.log1p())

    def expm1(self):
        """
        Computes exp(x-1) element-wise.
        :return: Matrix instance.
        """
        return self._new_instance(self.matrix.expm1())

    def min(self, axis=None):
        """
        The minimum value in matrix. If axis is not None, each minimum value along the axis.
        :param axis: None, 0, or 1.
        :return: Minimum value(s)
        """
        if axis is None:
            return self.matrix.min(axis)
        elif axis == 0:
            return self._new_instance(self.matrix.min(axis), row2word=[''])
        elif axis == 1:
            return self._new_instance(self.matrix.min(axis), col2word=[''])
        else:
            raise ValueError("Axis can only be None, 1, or 0")

    def max(self, axis=None):
        """
        The maximum value in matrix. If axis is not None, each maximum value along the axis.
        :param axis: None, 0, or 1.
        :return: Maximum value(s)
        """
        if axis is None:
            return self.matrix.max(axis)
        elif axis == 0:
            return self._new_instance(self.matrix.max(axis), row2word=[''])
        elif axis == 1:
            return self._new_instance(self.matrix.max(axis), col2word=[''])
        else:
            raise ValueError("Axis can only be None, 1, or 0")

    def mean(self, axis=None):
        """
        Average the matrix over the given axis. If the axis is None,
        average over both rows and columns, returning a scalar.
        :param axis: 0, 1, or None
        :return: Matrix instance or double, if axis is None
        """
        if axis is None:
            return self.matrix.mean(axis)
        elif axis == 0:
            return self._new_instance(self.matrix.mean(axis), row2word=[''])
        elif axis == 1:
            return self._new_instance(self.matrix.mean(axis), col2word=[''])
        else:
            raise ValueError("Axis can only be None, 1, or 0")

    def add_indices(self, indices, axis):
        """
        Add new indices to the matrix given an axis, filled with zeroes.
        If an index value is already present it is ignored.
        :param indices: List of words
        :param axis: 0 (rows) or 1 (cols)
        :return: Matrix with new index.
        """
        if axis == 0:
            new_indices = list(set(indices) - set(self.row2word))
            if len(new_indices) == 0:
                return self

            shape = (len(new_indices), self.shape[1])
            mat = self._new_instance(sp.coo_matrix(shape), row2word=new_indices)
        elif axis == 1:
            new_indices = list(set(indices) - set(self.col2word))

            if len(new_indices) == 0:
                return self

            shape = (self.shape[0], len(new_indices))
            mat = self._new_instance(sp.coo_matrix(shape), col2word=new_indices)
        else:
            raise ValueError("Axis can only be 0 or 1")

        return self.append(mat, axis)

    def merge(self, matrix, merge_function='add'):
        """
        Merge self and matrix such that the resulting matrix is a combination of both.
        This takes into consideration the row and col index of both matrices, such that a row-col index appearing
        in both matrices will be combined together. Duplicate entries will be combined by the given function,
        defaulted to adding them together. No assumption about the word order can be made after merge.
        :param matrix: Matrix to merge self with.
        :param merge_function: Function which to use for combining the values. One of: add, multiply, subtract.
                               Or pass an actual function which takes two Matrix instances as arguments.
        :return: Merged matrix.
        """
        if self.shape == (0, 0):
            return matrix
        elif matrix.shape == (0, 0):
            return self
            
        this_col2word = set(self.col2word)
        this_row2word = set(self.row2word)
        that_col2word = set(matrix.col2word)
        that_row2word = set(matrix.row2word)

        mat1 = self.add_indices(that_row2word, axis=0).add_indices(that_col2word, axis=1)
        mat2 = matrix.add_indices(this_row2word, axis=0).add_indices(this_col2word, axis=1)
        mat1 = mat1.synchronize_word_order(mat2)
        if isinstance(merge_function, str):
            return getattr(mat1, merge_function)(mat2)
        elif callable(merge_function):
            return merge_function(mat1, mat2)
        else:
            raise TypeError("merge_function has to be either string or function.")

    def multiply(self, factor):
        """
        Element-wise multiplication.
        :param factor:
        :return:
        """
        if isinstance(factor, IndexMatrix):
            if factor.shape == (1,1):
                return self.multiply(factor[0,0])
            elif factor.shape[0] == 1:
                diag = self._new_instance(sp.dia_matrix((factor.to_ndarray(), [0]),
                                                        shape=(factor.shape[1], factor.shape[1])),
                                                        row2word=factor.col2word)
                return self.dot(diag)
            elif factor.shape[1] == 1:
                diag = self._new_instance(sp.dia_matrix((factor.transpose().to_ndarray(), [0]),
                                          shape=(factor.shape[0], factor.shape[0])),
                                          col2word=factor.row2word)
                return diag.dot(self)
            else:
                return self._new_instance(self.matrix.multiply(factor.matrix))
        elif isinstance(factor, Number):
            return self._new_instance(self.matrix.multiply(factor))

    def negate(self):
        """
        Negation of all non-zero matrix values.
        :return: Negated matrix instance.
        """
        return self._new_instance(-self.matrix)

    def add(self, term):
        """
        Matrix and scalar addition.
        NOTE: Scalar addition does not affect zero values. That is, 0+1 = 0.
        If you insist on correct matrix addition, use matrices rather than scalars.
        :param term: To be added.
        :return: Resulted added matrix.
        """
        if isinstance(term, IndexMatrix):
            if term.shape[0] == 1:
                # Thank you, stack overflow
                # http://stackoverflow.com/questions/20060753/efficiently-subtract-vector-from-matrix-scipy
                mat = self.matrix.tocsc()
                t_mat = term.matrix
                mat.data += np.repeat(t_mat.toarray()[0], np.diff(mat.indptr))
                return self._new_instance(mat)
            elif term.shape[1] == 1:
                mat = self.matrix.copy()
                t_mat = term.matrix.T
                mat.data += np.repeat(t_mat.toarray()[0], np.diff(mat.indptr))
                return self._new_instance(mat)
            else:
                return self._new_instance(self.matrix + term.matrix)
        elif isinstance(term, Number):
            mat = self.matrix.copy()
            mat.data = mat.data + term
            return self._new_instance(mat)

    def subtract(self, term):
        """
        Matrix and scalar subtraction.
        NOTE: Scalar subtraction does not affect zero values. That is, 0-1 = 0.
        If you insist on correct matrix subtraction, use matrices rather than scalars.
        :param term: To be subtracted.
        :return: Resulted subtracted matrix.
        """
        return self.add(-term)

    def divide(self, factor):
        """
        Scalar and matrix division. Scalar division is equivalent to 1/factor * matrix.
        Note: Division by zero will return zero, rather than NaN.
        :param factor: Denominator to use for division.
        :return: Resulted divided matrix.
        """
        if isinstance(factor, Number):
            mat = self.matrix.copy()
            mat.data = mat.data / factor
            return self._new_instance(mat)
        elif isinstance(factor, IndexMatrix):
            if factor.shape[0] == 1:
                inverted = 1/factor
                length = factor.shape[1]
                diag = self._new_instance(sp.dia_matrix((inverted.to_ndarray(), [0]), shape=(length, length)),
                                          row2word=self.col2word)
                return self.dot(diag)
            elif factor.shape[1] == 1:
                inverted = 1/factor.transpose()
                length = factor.shape[0]
                diag = self._new_instance(sp.dia_matrix((inverted.to_ndarray(), [0]), shape=(length, length)),
                                          col2word=self.row2word)
                return diag.dot(self)
            else:
                return self.multiply(1/factor)
        else:
            raise TypeError("Has to be either scalar or of type Matrix")


    def rdivide(self, numerator):
        """
        element-wise division, where each matrix element works as the denominator.
        :param numerator: Numerator of the element-wise division.
        :return: Resulted matrix.
        """
        if isinstance(numerator, Number):
            mat = self.matrix.copy()
            mat.data = numerator / mat.data
            return self._new_instance(mat)
        else:
            raise TypeError("Has to be either scalar or of type Matrix")

    def sqrt(self):
        """
        element-wise square-root.
        :return: Squared matrix.
        """
        return self._new_instance(self.matrix.sqrt())

    def inverse(self):
        """
        Computes the inverse of a matrix.
        :return: The inverse of self.
        """
        return self._new_instance(sp.linalg.inv(self.matrix.tocsc()))

    def norm(self, axis=None):
        """
        Computes the norm of a matrix.
        :param axis: If None, computes the norm of the matrix.
                     If 0, computes the norm along the rows.
                     If 1, computes the norm along the columns.
        :return: The norm of a matrix, or the norm along an axis.
        """
        if axis not in (0, 1, None):
            raise ValueError("Axis must be 0, 1 or None")

        if axis is None:
            return np.linalg.norm(self.matrix.data)

        summed = self.multiply(self).sum(axis).sqrt()

        return summed

    def transpose(self):
        """
        Computes the transpose of a matrix.
        :return: The transpose.
        """
        return self._new_instance(self.matrix.transpose(), row2word=self.col2word, col2word=self.row2word)

    @property
    def word2row(self):
        """
        A dict of words along the row axis, containing the index of which row that correlates to which word.
        This is only generated when necessary.
        :return: Dict.
        """
        if self._word2row is None:
            self._word2row = {w: i for i, w in enumerate(self.row2word)}

        return self._word2row

    @property
    def word2col(self):
        """
        A dict of words along the col axis, containing the index of which col that correlates to which word.
        This is only generated when necessary.
        :return: Dict.
        """
        if self._word2col is None:
            self._word2col = {w: i for i, w in enumerate(self.col2word)}

        return self._word2col

    @property
    def row2word(self):
        """
        A list of words, where the index of a word correlates to the index of the row.
        :return: List.
        """
        return self._row2word

    @row2word.setter
    def row2word(self, row2word):
        if len(row2word) != len(self._row2word):
            raise ValueError("Length mismatch")

        self._row2word = row2word
        self._word2row = None

    @property
    def col2word(self):
        """
        A list of words, where the index of a word correlates to the index of the column.
        :return: List.
        """
        return self._col2word

    @col2word.setter
    def col2word(self, col2word):
        if len(col2word) != len(self._col2word):
            raise ValueError("Length mismatch")

        self._col2word = col2word
        self._word2col = None

    def _axis2words(self, arg, axis):
        """
        Returns a word representation of a given axis.
        :param arg: int, str, slice, or list.
                    If integer, return the corresponding word as list.
                    If string, return itself as list.
                    If slice, return the corresponding word list.
                    If list of integers and strings, return list with indices to words.
                    If list of booleans, return word list with all False elements removed.
        :param axis: 0 (rows) or 1 (cols)
        :return: Word list representation of given axis.
        """
        res = []
        blank2word = []

        if isinstance(arg, IndexMatrix):
            if axis == 0:
                return arg.row2word
            elif axis == 1:
                return arg.col2word
            else:
                raise ValueError("Axis must be 0 or 1.")

        if axis == 0:
            blank2word = self.row2word
        elif axis == 1:
            blank2word = self.col2word
        else:
            raise ValueError("Axis must be 0 or 1.")

        if isinstance(arg, slice):
            res = blank2word[arg]
        elif isinstance(arg, str):
            res = [arg]
        elif isinstance(arg, Integral):
            res = [blank2word[arg]]
        elif type(arg) in (list, np.ndarray):
            for i, content in enumerate(arg):
                if type(content) in (bool, np.bool_) and content:  # Convert boolean lists to words
                    res.append(blank2word[i])
                elif isinstance(content, str):
                    res.append(content)
                elif isinstance(content, Integral) and not isinstance(content, bool):  # Convert indices to words. The
                    res.append(blank2word[content])
        else:
            raise TypeError("This type is not supported for indexing")
        return res

    def _axis2indices(self, arg, axis):
        """
        Makes arg ready to perform slicing on a scipy matrix.
        :param arg: A string, int, list of string and/or int, or slice
        :param axis:
        :return:
        """

        if isinstance(arg, IndexMatrix):
            if axis == 0:
                return self._axis2indices(arg.row2word, axis=axis)
            elif axis == 1:
                return self._axis2indices(arg.col2word, axis=axis)
            else:
                raise ValueError("Axis must be 0 or 1.")

        res = []
        if axis == 0:
            word2blank = self.word2row
        elif axis == 1:
            word2blank = self.word2col
        else:
            raise ValueError("Axis needs to be 0 or 1")

        if isinstance(arg, slice):
            res = list(range(len(word2blank))[arg])
        elif isinstance(arg, str):
            res = word2blank[arg]
        elif isinstance(arg, Integral):
            res = arg
        elif type(arg) in (list, np.ndarray):
            for i, content in enumerate(arg):
                if type(content) in (bool, np.bool_) and content:  # Convert boolean lists to indices
                    res.append(i)
                elif isinstance(content, str):
                    res.append(word2blank[content])
                elif isinstance(content, Integral) and not isinstance(content, bool):
                    res.append(content)
        else:
            raise TypeError("This type is not supported for indexing")

        if type(arg) in (list, np.ndarray) and len(res) == 1:
            return res[0]
        else:
            return res

    @property
    def column(self):
        """
        Property for returning column.
        """
        class col(object):
            def __getitem__(_, item):
                return self[:,item]
            def __repr__(_):
                return self.__repr__()

        return col()

    @property
    def row(self):
        """
        Property for returning row.
        """
        return self

    def is_boolean(self):
        return  self.matrix.dtype == np.bool

    def get_value(self, arg):
        """
        Equivalent of calling the matrix directly, e.g. self[arg].
        :param arg: Words, integers, lists or slices. If tuple, it represents rows and columns respectively.
        :return: If a tuple of strings or integers, it returns the value in its place. Otherwise a sliced matrix.
        """
        # If matrix is boolean, return true values
        if isinstance(arg, IndexMatrix) and arg.is_boolean():
            return self * arg

        indices = [None, None]
        if self.matrix.format != 'csr':
            self.matrix = self.matrix.tocsr()

        # Get axes
        col2word = self.col2word
        if isinstance(arg, tuple):
            row2word = self._axis2words(arg[0], axis=0)
            col2word = self._axis2words(arg[1], axis=1)
        else:
            row2word = self._axis2words(arg, axis=0)

        # Get indices
        if isinstance(arg, tuple) and len(arg) == 2:
            indices[0] = self._axis2indices(arg[0], axis=0)
            indices[1] = self._axis2indices(arg[1], axis=1)
            indices = tuple(indices)
        else:
            indices = self._axis2indices(arg, axis=0)

        # some weird bug makes it impossible to use a tuple of lists as input. workaround.
        if isinstance(arg, tuple) and len(arg) == 2 \
                and type(indices[0]) in (np.ndarray, list) \
                and type(indices[1]) in (np.ndarray, list):
            res = self.matrix[indices[0], :][:, indices[1]]
        else:
            res = self.matrix[indices]

        if isinstance(res, Number):
            return res
        elif isinstance(res, sp.spmatrix):
            return self._new_instance(res, row2word=row2word, col2word=col2word)
        else:
            raise RuntimeError("Hm, we shouldn't have gotten this far...")

    def _new_instance(self, mat, row2word=None, col2word=None):
        """
        Returns a new instance of a matrix. If row2word or col2word are None, use the given one of its instance.
        :param mat: Scipy sparse matrix.
        :param row2word: List of word row indices.
        :param col2word: List of word column indices.
        :return: A Matrix instance
        """
        if row2word is None:
            row2word = self.row2word
        if col2word is None:
            col2word = self.col2word
        if not sp.issparse(mat):
            mat = sp.csr_matrix(mat)

        return IndexMatrix(mat, row2word=row2word, col2word=col2word)

    def __getitem__(self, arg):
        return self.get_value(arg)

    def __add__(self, term):
        return self.add(term)

    def __radd__(self, term):
        return self.add(term)

    def __sub__(self, term):
        return self.subtract(term)

    def __rsub__(self, term):
        return self.subtract(term)

    def __neg__(self):
        return self.negate()

    def __mul__(self, factor):
        return self.multiply(factor)

    def __rmul__(self, factor):
        return self.multiply(factor)

    def __div__(self, denominator):
        return self.divide(denominator)

    def __rdiv__(self, numerator):
        return self.rdivide(numerator)

    def __truediv__(self, denominator):
        return self.divide(denominator)

    def __rtruediv__(self, numerator):
        return self.rdivide(numerator)

    def __iter__(self):
        """
        Row-wise iteration.
        :return: Iterator.
        """
        for row in range(self.shape[0]):
            yield self[row]


    def __eq__(self, other):
        if isinstance(other, IndexMatrix)\
                and self.matrix.shape == other.matrix.shape\
                and self.row2word == other.row2word\
                and self.col2word == other.col2word\
                and (self.matrix != other.matrix).getnnz() == 0:
            return True
        elif isinstance(other, Number):
            return self._new_instance(self.matrix.__eq__(other).astype(np.bool))
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, IndexMatrix) \
                and (self.matrix.shape != other.matrix.shape\
                or self.row2word != other.row2word\
                or self.col2word != other.col2word\
                or (self.matrix != other.matrix).getnnz() > 0):
            return True
        elif isinstance(other, Number):
            return self._new_instance(self.matrix.__ne__(other).astype(np.bool))
        else:
            return False


    def __ge__(self, other):
        if isinstance(other, IndexMatrix):
            other = other.matrix
        return self._new_instance(self.matrix.__ge__(other).astype(np.bool))

    def __gt__(self, other):
        if isinstance(other, IndexMatrix):
            other = other.matrix
        return self._new_instance(self.matrix.__gt__(other).astype(np.bool))

    def __le__(self, other):
        if isinstance(other, IndexMatrix):
            other = other.matrix
        return self._new_instance(self.matrix.__le__(other).astype(np.bool))

    def __lt__(self, other):
        if isinstance(other, IndexMatrix):
            other = other.matrix
        return self._new_instance(self.matrix.__lt__(other).astype(np.bool))

    def __repr__(self):
        """
        The representation of a Matrix is a table with row and column words printed out.

        Example:

             hey good  bye
        hey  1.0  2.0  3.0
        good 4.0  5.0  6.0
        bye  7.0  8.0  9.0
        :return: A string representation of the matrix.
        """
        return self.print_matrix(n_rows=self._display_max_rows, n_cols=self._display_max_cols)

    def __str__(self):
        return self.__repr__()

    def to_coo(self):
        """
        Return a scipy.sparse.coo representation of the matrix
        :return: Matrix of type scipy.sparse.coo
        """
        return self.matrix.tocoo()

    def to_ndarray(self):
        """
        Return a dense numpy.matrix representation of the matrix
        :return: Matrix of type numpy.matrix
        """
        return self.matrix.todense()


    def to_dataframe(self):
        """
        Return a dense numpy.matrix representation of the matrix
        :return: Matrix of type numpy.matrix
        """
        import pandas
        return pandas.DataFrame(self.to_ndarray(), columns=self.col2word, index=self.row2word)


    def print_matrix(self, n_rows=None, n_cols=None):
        """
        Prints n_rows and n_cols of the matrix. If either is set to None, print the standard amount.
        :param n_rows:
        :param n_cols:
        :return: String representation of the matrix.
        """
        # We need to get it to a list of list representation for tabulate

        if n_cols is None:
            n_cols = self._display_max_cols
        if n_rows is None:
            n_rows = self._display_max_rows
        n_rows = min(n_rows, self.shape[0])
        n_cols = min(n_cols, self.shape[1])
        mat = self.matrix[:n_rows, :n_cols]

        # Add column words
        res = [[str(list(self.shape))] + self.col2word[:n_cols]]

        # Add row words
        res += [[self.row2word[i]] + row for i, row in enumerate(mat.todense().tolist())]

        # Add ellipses
        if n_cols < self.shape[1]:
            [row.append("...") for row in res]
        if n_rows < self.shape[0]:
            res.append(["..."] * len(res[0]))

        table = tabulate(res, tablefmt='plain')
        return table

__all__ = ['IndexMatrix']
