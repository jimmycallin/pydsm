# -*- coding: utf-8 -*-
from pydsm.matrix import Matrix
import pydsm.model


def _assure_consistency(dsm, vector):
    if not vector.is_vector():
        raise ValueError("Vector should have the shape (1,n).")

    vector = vector.synchronize_word_order(dsm, axis=1)
    if dsm.col2word != vector.col2word:
        raise ValueError("Columns of each matrix must match")
    return vector

def dot(dsm, vector):
    """
    Calculate dot product distance for all words in dsm against vector.
    """
    vector = _assure_consistency(dsm, vector)
    return dsm.matrix.dot(vector.transpose())


def euclidean(dsm, vector):
    """
    Calculate inversed euclidean distance for all words in dsm against vector.
    """
    vector = _assure_consistency(dsm, vector)
    return 1/(1+dsm.matrix.subtract(vector).norm(axis=1))


def cos(dsm, vector):
    """
    Calculate cosine distance for all words in dsm against all words in vector.
    """
    vector = _assure_consistency(dsm, vector)

    if isinstance(dsm, pydsm.model.DSM):
        mat1 = dsm.matrix
    elif isinstance(dsm, Matrix):
        mat1 = dsm

    dotted = mat1.dot(vector.transpose())
    mat1_norms = mat1.multiply(mat1).sum(axis=1).sqrt()
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
        scores.append(simfunc(dsm, row).sort(key='sum', axis=0, ascending=False))


    res = scores[0]
    for i in scores[1:]:
        res = res.append(i, axis=1)
    return res

__dsm__ = ['nearest_neighbors']
