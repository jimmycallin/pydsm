__author__ = 'jimmy'

import unittest
import pydsm.similarity as similarity
from pydsm import IndexMatrix
import scipy.sparse as sp

class TestSimilarity(unittest.TestCase):
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

    def test_cos(self):
        res = [[1.0000000000000002], [0.5732594911200148]]
        self.assertEqual(similarity.cos(self.mat, self.mat[0]).to_ndarray().tolist(), res)

    def test_dot(self):
        res = [[38.0], [32.0]]
        self.assertEqual(similarity.dot(self.mat, self.mat[0]).to_ndarray().tolist(), res)

    def test_euclidean(self):
        res = [[0.12178632452799958], [0.0]]
        self.assertEqual(similarity.euclidean(self.mat, self.mat[0]).to_ndarray().tolist(), res)


if __name__ == '__main__':
    unittest.main()
