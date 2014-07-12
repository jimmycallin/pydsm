
def epmi(matrix):
    """
    Exponential pointwise mutual information
    """
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    total = row_sum.sum(axis=0)[0,0]
    inv_col_sum = total / col_sum  # shape (1,n)
    inv_row_sum = total / row_sum      # shape (n,1)

    mat = matrix / total
    mat = mat * inv_row_sum
    mat = mat * inv_col_sum
    return mat


def pmi(matrix):
    """
    Pointwise mutual information
    """
    mat = epmi(matrix).log1p()
    return mat


def ppmi(matrix):
    """
    Positive pointwise mutual information
    """
    mat = pmi(matrix)
    return mat[mat > 0]


def npmi(matrix):
    """
    Normalized pointwise mutual information
    """
    total = matrix.sum(axis=0).sum(axis=1)[0,0]
    log_probs = -matrix.divide(total).log()
    return pmi(matrix).divide(log_probs)


def pnpmi(matrix):
    """
    Positive normalized pointwise mutual information
    """
    mat = npmi(matrix)
    return mat[mat > 0]


def apply_weighting(dsm, weighting_func=ppmi):
    """
    Available functions: ppmi, plog, epmi, plmi.
    """

    return dsm._new_instance(weighting_func(dsm.matrix), add_to_config={'weighting': weighting_func})


__dsm__ = ['apply_weighting']
