import unittest
import pydsm.composition as composition
from pydsm import IndexMatrix
import scipy.sparse as sp

__author__ = 'jimmy'


class TestLinear_additive(unittest.TestCase):
    def create_mat(self, list_, row2word=None, col2word=None):
        if row2word is None:
            row2word = self.row2word
        if col2word is None:
            col2word = self.col2word

        return IndexMatrix(sp.coo_matrix(list_), row2word, col2word)

    def setUp(self):
        self.spmat = sp.coo_matrix([[2, 5, 3], [0, 1, 9]])
        self.row2word = ['a', 'b']
        self.col2word = ['furiously', 'makes', 'sense']
        self.mat = IndexMatrix(self.spmat, self.row2word, self.col2word)

    def test_linear_additive(self):
        res = self.create_mat([[0.2, 1.4, 8.4]], row2word=['a b'])
        self.assertEqual(res, composition.linear_additive(self.mat[0], self.mat[1], alpha=0.1, beta=0.9))

    def multiplicative(self):
        res = self.create_mat([[0, 0.45, 2.43]], row2word=['a b'])
        self.assertEqual(res, composition.multiplicative(self.mat[0], self.mat[1], alpha=0.1, beta=0.9))