# -*- coding: utf-8 -*-
import unittest

import numpy as np
import scipy.sparse as sp

from pydsm.indexmatrix import IndexMatrix


__author__ = 'Jimmy Callin'


class TestMatrix(unittest.TestCase):
    def create_mat(self, list_, row2word=None, col2word=None):
        if row2word is None:
            row2word = self.row2word
        if col2word is None:
            col2word = self.col2word

        return IndexMatrix(sp.coo_matrix(list_), row2word, col2word)

    def setUp(self):
        self.spmat = sp.coo_matrix([[3, 2, 1], [6, 5, 4], [7, 8, 9]])
        self.row2word = ['green', 'ideas', 'sleep']
        self.col2word = ['furiously', 'makes', 'sense']
        self.mat = IndexMatrix(self.spmat, self.row2word, self.col2word)


    def test_init(self):
        # One too few argument
        self.assertRaises(TypeError, IndexMatrix, self.spmat, self.row2word)
        # Col2word != column length
        self.assertRaises(ValueError, IndexMatrix, self.spmat, self.row2word, self.col2word[:1])
        # Row2word != column length
        self.assertRaises(ValueError, IndexMatrix, self.spmat, self.row2word[:1], self.col2word)
        # Dense matrices are converted to sparse
        dense = np.array([[1, 2, 3], [4, 5, 5], [7, 8, 9]])
        self.assertTrue(isinstance(IndexMatrix(dense, self.row2word, self.col2word).matrix, sp.spmatrix))
        # Dict to matrix
        m = {'a': {'ba': 3}, 'c': {'ba': 5, 'bc': 4}}
        res =  self.create_mat([[0, 3.0], [4.0, 5.0]],
                               row2word=['a', 'c'], col2word=['bc', 'ba'])
        mat = IndexMatrix(m)
        self.assertEqual(mat.synchronize_word_order(res, 0).synchronize_word_order(res, 1), res)


    def test_apply(self):
        res = self.create_mat([[8, 4, 2], [64, 32, 16], [128, 256, 512]])
        self.assertEqual(self.mat.apply(np.exp2), res)

    def test_sort(self):
        # Test row descending
        res1 = self.create_mat([[7.0, 8.0, 9.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
                               row2word=['sleep', 'ideas', 'green'])
        self.assertEqual(self.mat.sort('sum', axis=0, ascending=False), res1)
        # Test row ascending
        res2 = self.create_mat([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0], [7.0, 8.0, 9.0]])
        self.assertEqual(self.mat.sort('sum', axis=0, ascending=True), res2)
        # Test col ascending
        res3 = self.create_mat([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [9.0, 8.0, 7.0]],
                               col2word=['sense', 'makes', 'furiously'])
        self.assertEqual(self.mat.sort('sum', axis=1, ascending=True), res3)
        # Test col descending
        res4 = self.create_mat([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0], [7.0, 8.0, 9.0]])
        self.assertTrue(self.mat.sort('sum', axis=1, ascending=False), res4)


    def test_row_col_data(self):
        row, col, data = self.mat.row_col_data
        self.assertTrue((row == self.spmat.row).all()
                        and (col == self.spmat.col).all()
                        and (data == self.spmat.data).all())


    def test_delete(self):
        res = self.create_mat([[3, 2, 1], [7, 8, 9]], row2word=['green', 'sleep'])
        self.assertEqual(self.mat.delete('ideas', axis=0), res)


    def test_synchronise_word_order(self):
        res = self.create_mat([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [9.0, 8.0, 7.0]],
                              col2word=['sense', 'makes', 'furiously'])
        self.assertEqual(self.mat.synchronize_word_order(res, axis=1), res)

    def test_append(self):
        res = self.create_mat([[3, 2, 1], [6, 5, 4], [7, 8, 9], [1, 1, 1]], row2word=self.row2word + ['cool'])
        self.assertEqual(self.mat.append(self.create_mat([[1, 1, 1]], col2word=self.col2word, row2word=['cool']),
                                         axis=0), res)

    def test_is_vector(self):
        res = self.create_mat([[3, 2, 2]], row2word=['vector'])
        self.assertTrue(res.is_vector())
        self.assertFalse(self.mat.is_vector())

    def test_sum(self):
        res = self.create_mat([[16, 15, 14]], row2word=[''])
        self.assertEqual(self.mat.sum(axis=0), res)

    def test_dot(self):
        dotted = self.create_mat([[16, 15, 14]], row2word=[''])
        res = self.create_mat([[92.0], [227.0], [358.0]], col2word=[''])
        self.assertRaises(ValueError, self.mat.dot, dotted)
        self.assertEqual(self.mat.dot(dotted.transpose()), res)


    def test_shape(self):
        res = self.create_mat([[92.0], [227.0], [358.0]], col2word=[''])
        self.assertEqual(res.shape, (3, 1))
        self.assertEqual(self.mat.shape, (3, 3))

    def test_log(self):
        res = self.create_mat([[1.0986122886681098, 0.6931471805599453, 0.0],
                               [1.791759469228055, 1.6094379124341003, 1.3862943611198906],
                               [1.9459101490553132, 2.0794415416798357, 2.1972245773362196]])
        self.assertAlmostEqual(self.mat.log(), res)

    def test_log1p(self):
        res = self.create_mat([[1.3862943611198906, 1.0986122886681098, 0.6931471805599453],
                               [1.9459101490553132, 1.791759469228055, 1.6094379124341003],
                               [2.0794415416798357, 2.1972245773362196, 2.302585092994046]])

        self.assertAlmostEqual(self.mat.log1p(), res)

    def test_expm1(self):
        res = self.create_mat([[19.085536923187668, 6.38905609893065, 1.7182818284590453],
                               [402.4287934927351, 147.4131591025766, 53.598150033144236],
                               [1095.6331584284585, 2979.9579870417283, 8102.083927575384]])
        self.assertAlmostEqual(self.mat.expm1(), res)

    def test_min(self):
        res = self.create_mat([[1], [4], [7]], col2word=[''])
        self.assertEqual(self.mat.min(axis=1), res)
        self.assertEqual(self.mat.min(axis=None), 1.0)

    def test_max(self):
        res = self.create_mat([[3], [6], [9]], col2word=[''])
        self.assertEqual(self.mat.max(axis=1), res)
        self.assertEqual(self.mat.max(axis=None), 9.0)

    def test_mean(self):
        self.assertEqual(self.mat.mean(axis=None), 5.0)
        res = self.create_mat([[2.0], [5.0], [8.0]], col2word=[''])
        self.assertEqual(self.mat.mean(axis=1), res)

    def test_add_indices(self):
        res1 = self.create_mat([[3, 2, 1], [6, 5, 4], [7, 8, 9], [0, 0, 0]], row2word=self.row2word + ['möh'])
        appended1 = self.create_mat([[0, 0, 0]], row2word=['möh'])
        self.assertEqual(self.mat.append(appended1, axis=0), res1)

        res2 = self.create_mat([[3, 2, 1, 0], [6, 5, 4, 0], [7, 8, 9, 0]], col2word=self.col2word + ['meh'])
        appended2 = self.create_mat([[0], [0], [0]], col2word=['meh'])
        self.assertEqual(self.mat.append(appended2, axis=1), res2)

    def test_merge(self):
        res = self.create_mat([[6, 4, 2, 0], [12, 10, 8, 0], [14, 16, 18, 0]], col2word=self.col2word + ['meh'])
        mergmat = self.create_mat([[3, 2, 1, 0], [6, 5, 4, 0], [7, 8, 9, 0]], col2word=self.col2word + ['meh'])
        self.assertEqual(self.mat.merge(mergmat, merge_function='add'), res)

    def test_multiply(self):
        res = self.create_mat([[4.5, 3.0, 1.5], [9.0, 7.5, 6.0], [10.5, 12.0, 13.5]])
        self.assertEqual(self.mat.multiply(1.5), res)
        self.assertEqual(self.mat * 1.5, res)
        self.assertEqual(1.5 * self.mat, res)


    def test_negate(self):
        res = self.create_mat([[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0], [-7.0, -8.0, -9.0]])
        self.assertEqual(-self.mat, res)

    def test_add(self):
        added = self.create_mat([[1.5, 1.0, 0.5], [3.0, 2.5, 2.0], [3.5, 4.0, 4.5]])
        res = self.create_mat([[4.5, 3.0, 1.5], [9.0, 7.5, 6.0], [10.5, 12.0, 13.5]])
        self.assertEqual(self.mat.add(added), res)
        self.assertEqual(self.mat + added, res)
        self.assertEqual(added + self.mat, res)

        another = self.create_mat([[1, 2], [4, 5]], row2word=['no', 'idea'], col2word=['er', 'sdf'])
        again = self.create_mat([[3, 4], [6, 7]], row2word=['no', 'idea'], col2word=['er', 'sdf'])
        self.assertEqual(another + 2, again)

    def test_subtract(self):
        res = self.create_mat([[1.5, 1.0, 0.5], [3.0, 2.5, 2.0], [3.5, 4.0, 4.5]])
        added = self.create_mat([[4.5, 3.0, 1.5], [9.0, 7.5, 6.0], [10.5, 12.0, 13.5]])
        self.assertEqual(added.subtract(self.mat), res)
        self.assertEqual(added - self.mat, res)

        again = self.create_mat([[1, 2], [4, 5]], row2word=['no', 'idea'], col2word=['er', 'sdf'])
        another = self.create_mat([[3, 4], [6, 7]], row2word=['no', 'idea'], col2word=['er', 'sdf'])
        self.assertEqual(another - 2, again)

        # test vector subtraction
        res2 = self.create_mat([[3.5, 1.0, 0.5], [8.0, 5.5, 5.0], [9.5, 10.0, 12.5]])
        self.assertEqual(res2, added - self.create_mat([[1, 2, 1]], row2word=['1']))
        res3 = self.create_mat([[1.5, -1.0, -1.5], [3.0, 0.5, 0], [6.5, 7.0, 9.5]])
        self.assertEqual(res3, res2 - self.create_mat([[2], [5], [3]], col2word=['a']))

    def test_std(self):
        res = self.create_mat([[0.816496580927726], [0.816496580927726], [0.816496580927726]], col2word=[''])
        self.assertEqual(self.mat.std(1), res)


    def test_divide(self):
        res = self.create_mat([[1.5, 1.0, 0.5], [3.0, 2.5, 2.0], [3.5, 4.0, 4.5]])
        div = self.create_mat([[3, 2, 1], [6, 5, 4], [7, 8, 9]])
        v = div[0]
        another = self.create_mat([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        self.assertEqual(div / 2, res)
        self.assertEqual(div.divide(2), res)
        self.assertEqual(div.divide(another), res)
        zeroes = self.create_mat([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        # test that 0 multiplication does not equal NaN
        self.assertFalse(np.isnan(zeroes.divide(zeroes).matrix.todense()).any())

        # test vector division
        vector_division = div.divide(v)
        res2 = self.create_mat([[1.0, 1.0, 1.0], [2.0, 2.5, 4.0], [2.333333333333333, 4.0, 9.0]])
        self.assertEqual(vector_division, res2)
        res3 = self.create_mat([[1.0, 0.6666666666666666, 0.3333333333333333],
                                [3.0, 2.5, 2.0],
                                [7.0, 8.0, 9.0]])
        self.assertEqual(res3, div.divide(v.transpose()))


    def test_rdivide(self):
        res = self.create_mat([[1.5, 1.0, 0.5], [3.0, 2.5, 2.0], [3.5, 4.0, 4.5]])
        div = self.create_mat([[2.6666666666666665, 4.0, 8.0],
                               [1.3333333333333333, 1.6, 2.0],
                               [1.1428571428571428, 1.0, 0.8888888888888888]])
        self.assertEqual(4 / res, div)

    def test_sqrt(self):
        res = self.create_mat([[1.7320508075688772, 1.4142135623730951, 1.0],
                               [2.449489742783178, 2.23606797749979, 2.0],
                               [2.6457513110645907, 2.8284271247461903, 3.0]])
        self.assertEqual(self.mat.sqrt(), res)

    def test_inverse(self):
        # Is singular matrix
        self.assertRaises(RuntimeError, self.mat.inverse)
        invertible = self.create_mat([[-4.5, 3.0, 1.5], [9.0, 7.5, 6.0], [10.5, 12.0, 13.5]])

        res = self.create_mat([[-0.11111111111111116, 0.08547008547008546, -0.025641025641025654],
                               [0.2222222222222222, 0.29059829059829057, -0.1538461538461538],
                               [-0.11111111111111106, -0.32478632478632474, 0.23076923076923073]])

        self.assertEqual(invertible.inverse(), res)

    def test_norm(self):
        another = self.create_mat([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        self.assertEqual(another.norm(), 6)

    def test_transpose(self):
        res = self.create_mat([[3.0, 6.0, 7.0], [2.0, 5.0, 8.0], [1.0, 4.0, 9.0]],
                              row2word=self.col2word, col2word=self.row2word)
        self.assertEqual(self.mat.transpose(), res)

    def test_word2row(self):
        res = {'green': 0, 'ideas': 1, 'sleep': 2}
        self.assertEqual(self.mat.word2row, res)

    def test_word2col(self):
        res = {'furiously': 0, 'makes': 1, 'sense': 2}
        self.assertEqual(self.mat.word2col, res)

    def test_row2word(self):
        self.assertEqual(self.mat.row2word, ['green', 'ideas', 'sleep'])

    def test_col2word(self):
        self.assertEqual(self.mat.col2word, ['furiously', 'makes', 'sense'])

    def test__axis2words(self):
        self.assertEqual(self.mat._axis2words(0, axis=0), ['green'])
        self.assertEqual(self.mat._axis2words(2, axis=0), ['sleep'])
        self.assertEqual(self.mat._axis2words(1, axis=1), ['makes'])
        self.assertEqual(self.mat._axis2words(2, axis=1), ['sense'])
        self.assertEqual(self.mat._axis2words([2, 1], axis=1), ['sense', 'makes'])
        self.assertEqual(self.mat._axis2words([2], axis=1), ['sense'])
        self.assertEqual(self.mat._axis2words([True, True, False], axis=1), ['furiously', 'makes'])
        self.assertEqual(self.mat._axis2words(np.array([True, True, False]), axis=1), ['furiously', 'makes'])
        self.assertEqual(self.mat._axis2words(['green', 'sleep', 'ideas'], axis=0), ['green', 'sleep', 'ideas'])


    def test__axis2indices(self):
        self.assertEqual(self.mat._axis2indices(0, axis=0), 0)
        self.assertEqual(self.mat._axis2indices(2, axis=0), 2)
        self.assertEqual(self.mat._axis2indices(1, axis=1), 1)
        self.assertEqual(self.mat._axis2indices(2, axis=1), 2)
        self.assertEqual(self.mat._axis2indices([2, 1], axis=1), [2, 1])
        self.assertEqual(self.mat._axis2indices([2], axis=1), 2)
        self.assertEqual(self.mat._axis2indices([True, True, False], axis=1), [0, 1])
        self.assertEqual(self.mat._axis2indices(np.array([True, True, False]), axis=1), [0, 1])
        self.assertEqual(self.mat._axis2indices(['green', 'sleep', 'ideas'], axis=0), [0, 2, 1])
        self.assertEqual(self.mat._axis2indices(['green', 2, 'ideas'], axis=0), [0, 2, 1])

    def test_get_value(self):
        self.assertEqual(self.mat[:, :], self.mat)

        res1 = self.create_mat([[3], [6], [7]], col2word=['furiously'])
        self.assertEqual(self.mat[:, 0], res1)

        res2 = self.create_mat([[3, 2], [6, 5], [7, 8]], col2word=['furiously', 'makes'])
        self.assertEqual(self.mat.get_value((slice(0, len(self.row2word), 1), slice(0, 2, 1))), res2)

        res3 = self.create_mat([[3, 1], [7, 9], [6, 4]], col2word=['furiously', 'sense'],
                               row2word=['green', 'sleep', 'ideas'])
        self.assertEqual(self.mat[[0, 2, 1], [0, 2]], res3)

        self.assertEqual(self.mat[self.mat], self.mat)
        self.assertEqual(self.mat[self.mat[:10]], self.mat[:10])

        # Test inequality
        res4 = self.create_mat([[True, False], [True, True], [True, True]], col2word=['furiously', 'sense'],
                               row2word=['green', 'sleep', 'ideas'])
        self.assertEqual(res3 > 2, res4)

        res5 = self.create_mat([[3, 0], [7, 9], [6, 4]], col2word=['furiously', 'sense'],
                               row2word=['green', 'sleep', 'ideas'])
        self.assertEqual(res5, res3[res3 > 2])

        self.assertRaises(IndexError, self.mat.get_value, 5)

    def test__new_instance(self):
        self.assertEqual(self.mat, self.mat._new_instance(self.spmat, row2word=self.row2word, col2word=self.col2word))

    def test_to_coo(self):
        self.assertTrue((self.mat.to_coo() != self.spmat).getnnz() == 0)

    def test_to_ndarray(self):
        res = np.array([[3., 2., 1.],
                        [6., 5., 4.],
                        [7., 8., 9.]])
        self.assertTrue((self.mat.to_ndarray() == res).all())

    def test_print_matrix(self):
        res = '[3, 3]  furiously  makes  sense\ngreen   3.0        2.0    1.0\nideas   6.0        5.0    4.0\nsleep   7.0        8.0    9.0'
        self.assertEqual(self.mat.print_matrix(), res)
        res2 = '[3, 3]  furiously  makes  ...\ngreen   3.0        2.0    ...\nideas   6.0        5.0    ...\n...     ...        ...    ...'
        self.assertEqual(self.mat.print_matrix(n_rows=2, n_cols=2), res2)


if __name__ == '__main__':
    unittest.main()