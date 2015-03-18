# -*- coding: utf-8 -*-
from pydsm.indexmatrix import IndexMatrix
import pydsm.model


def _assure_consistency(matrix, vector):
    return vector.synchronize_word_order(matrix, axis=1)

def dot(matrix, vector, assure_consistency=False):
    """
    Calculate dot product distance for all words in matrix against vector.
    """
    if assure_consistency:
        vector = _assure_consistency(matrix, vector)
    return matrix.dot(vector.transpose())


def euclidean(matrix, vector, assure_consistency=False):
    """
    Calculate inversed euclidean distance for all words in matrix against vector.
    """
    if assure_consistency:
        vector = _assure_consistency(matrix, vector)
    inv_euc= 1/(1+matrix.subtract(vector).norm(axis=1))
    return inv_euc.sort(ascending=False)


def cos(mat1, mat2, assure_consistency=False):
    """
    Calculate cosine distance for all words in matrix against all words in second matrix.
    Params:
        mat1: A matrix to check all values against
        mat2: Another matrix.
        assure_consistency: If set, it makes sure the matrix and vector share  the same column indices. 
                            This makes it more secure, but a bit slower.
    """

    if assure_consistency:
        mat2 = _assure_consistency(mat1, mat2)

    if mat1.is_vector() and mat2.is_vector():
        return _vector_vector_cos(mat1, mat2)
        

    mat1 = mat1 / mat1.norm(axis=1)
    mat2 = mat2 / mat2.norm(axis=1)
    dotted = mat1.dot(mat2.transpose())
    return dotted

def _vector_vector_cos(v1, v2, assure_consistency=False):
    """
    Faster calculation for vector pair cosine similarity.
    """
    return v1.dot(v2.transpose()) / (v1.norm() * v2.norm())


__dsm__ = ['nearest_neighbors']
