from unittest import TestCase

import scipy.sparse as sp

from pydsm import IndexMatrix
import pydsm.weighting as weighting


__author__ = 'jimmy'


class TestWeighting(TestCase):
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

    def test_epmi(self):
        res = self.create_mat([[2.0, 1.6666666666666667, 0.5], [0.0, 0.33333333333333337, 1.5]])
        self.assertEqual(weighting.epmi(self.mat), res)

    def test_pmi(self):
        res = self.create_mat([[0.6931471805599453, 0.5108256237659907, -0.6931471805599453],
                               [0.0, -1.0986122886681096, 0.4054651081081644]])
        self.assertEqual(res, weighting.pmi(self.mat))

    def test_ppmi(self):
        res = self.create_mat([[0.6931471805599453, 0.5108256237659907, 0.0], [0.0, 0.0, 0.4054651081081644]])
        self.assertEqual(weighting.ppmi(self.mat), res)

    def test_npmi(self):
        res = self.create_mat([[0.3010299956639812, 0.3684827970831031, -0.3653681296292078],
                               [0.0, -0.3667257913420846, 0.507778585013894]])
        self.assertEqual(weighting.npmi(self.mat), res)

    def test_pnpmi(self):
        res = self.create_mat([[0.3010299956639812, 0.3684827970831031, 0.0], [0.0, 0.0, 0.507778585013894]])
        self.assertEqual(weighting.pnpmi(self.mat), res)
        
        