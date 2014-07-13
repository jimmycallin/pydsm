# -*- coding: utf-8 -*-
from pydsm.matrix import Matrix
import pydsm.model


def _assure_consistency(matrix, vector):
    if not vector.is_vector():
        raise ValueError("Vector should have the shape (1,n).")

    vector = vector.synchronize_word_order(matrix, axis=1)
    if matrix.col2word != vector.col2word:
        raise ValueError("Columns of each matrix must match")
    return vector

def dot(matrix, vector):
    """
    Calculate dot product distance for all words in matrix against vector.
    """
    vector = _assure_consistency(matrix, vector)
    return matrix.dot(vector.transpose())


def euclidean(matrix, vector):
    """
    Calculate inversed euclidean distance for all words in matrix against vector.
    """
    vector = _assure_consistency(matrix, vector)
    return 1/(1+matrix.subtract(vector).norm(axis=1))


def cos(matrix, vector):
    """
    Calculate cosine distance for all words in matrix against all words in vector.
    """
    vector = _assure_consistency(matrix, vector)


    dotted = matrix.dot(vector.transpose())
    mat1_norms = matrix.multiply(matrix).sum(axis=1).sqrt()
    mat2_norms = vector.multiply(vector).sum(axis=1).sqrt()
    mat1_mat2_norms = mat1_norms.multiply(mat2_norms)
    neighbors = dotted.multiply(1 / mat1_mat2_norms)

    return neighbors


def nearest_neighbors(dsm, arg, simfunc=cos):
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

__dsm__ = ['nearest_neighbors']
