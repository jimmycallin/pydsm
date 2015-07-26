from math import log

def epmi(matrix):
    """
    Exponential pointwise mutual information
    """
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    total = row_sum.sum(axis=0)[0, 0]
    inv_col_sum = 1 / col_sum  # shape (1,n)
    inv_row_sum = 1 / row_sum      # shape (n,1)
    inv_col_sum = inv_col_sum * total

    mat = matrix * inv_row_sum
    mat = mat * inv_col_sum
    return mat


def smoothed_epmi(matrix, alpha):
    """
    Performs smoothed epmi.
    See smoothed_ppmi for more info.
    Derived from this:

    #(w,c) / #(TOT)
    --------------
    (#(w) / #(TOT)) * (#(c)^a / #(TOT)^a)
    ==>
    #(w,c) / #(TOT)
    --------------
    (#(w) * #(c)^a) / #(TOT)^(a+1))
    ==>
    #(w,c)
    ----------
    (#(w) * #(c)^a) / #(TOT)^a
    ==>
    #(w,c) * #(TOT)^a
    ----------
    #(w) * #(c)^a
    """

    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0).power(alpha)
    total = row_sum.sum(axis=0).power(alpha)[0, 0]
    inv_col_sum = 1 / col_sum  # shape (1,n)
    inv_row_sum = 1 / row_sum      # shape (n,1)
    inv_col_sum = inv_col_sum * total

    mat = matrix * inv_row_sum
    mat = mat * inv_col_sum
    return mat


def pmi(matrix):
    """
    Pointwise mutual information
    """
    mat = epmi(matrix).log()
    return mat


def shifted_pmi(matrix, k):
    """
    Shifted pointwise mutual information
    """
    mat = pmi(matrix) - log(k)
    return mat


def smoothed_pmi(matrix, alpha):
    """
    Smoothed pointwise mutual information
    See smoothed_ppmi for more information.
    """
    mat = smoothed_epmi(matrix, alpha).log()
    return mat


def ppmi(matrix):
    """
    Positive pointwise mutual information
    """
    mat = pmi(matrix)
    return mat[mat > 0]


def shifted_ppmi(matrix, k):
    """
    Shifted positive pointwise mutual information
    """
    mat = shifted_pmi(matrix, k)
    return mat[mat > 0]


def smoothed_ppmi(matrix, alpha):
    """
    Smoothed positive pointwise mutual information

    Performs PPMI with context distribution smoothing,
    as described by [1].

    [1] Levy, Goldberg, Dagan (2015). Improving Distributional
        Similarity with Lessons Learned from Word Embeddings
    """
    mat = smoothed_pmi(matrix, alpha)
    return mat[mat > 0]


def npmi(matrix):
    """
    Normalized pointwise mutual information
    """
    total = matrix.sum(axis=0).sum(axis=1)[0, 0]
    log_probs = -matrix.divide(total).log()
    return pmi(matrix).divide(log_probs)


def pnpmi(matrix):
    """
    Positive normalized pointwise mutual information
    """
    mat = npmi(matrix)
    return mat[mat > 0]


def lmi(matrix):
    """
    Local mutual information (not tested).
    """
    ppmi_mat = ppmi(matrix)
    return ppmi_mat.multiply(matrix.log())


__dsm__ = ['apply_weighting']
