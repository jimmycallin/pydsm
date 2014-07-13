from pydsm.matrix import Matrix

def linear_additive(v1, v2, alpha=0.5, beta=0.5):
    """
    Weighted elementwise addition.
    """
    compword = str(v1.row2word[0]) + " " + str(v2.row2word[0])
    comp = (alpha * v1) + (beta * v2)
    comp.row2word = [compword]
    return comp


def multiplicative(v1, v2, alpha=1, beta=1):
    """
    Weighted elementwise multiplication.
    """
    compword = str(v1.row2word[0]) + " " + str(v2.row2word[0])
    comp = (alpha * v1) * (beta * v2)
    comp.row2word = [compword]
    return comp


def compose(dsm, w1, w2, compfunc=linear_additive, **kwargs):
    """
    Returns a space containing the distributional vector of a composed word pair.
    The composition type is decided by compfunc.
    """
    if isinstance(w1, str):
        w1_string = w1
        vector1 = dsm[w1]
    elif isinstance(w1, Matrix) and w1.is_vector():
        w1_string = w1.row2word[0]
        vector1 = w1

    if isinstance(w2, str):
        w2_string = w2
        vector2 = dsm[w2]
    elif isinstance(w2, Matrix) and w2.is_vector():
        w2_string = w2.row2word[0]
        vector2 = w2

    res_vector = compfunc(vector1, vector2, **kwargs)

    return res_vector



__dsm__ = ['compose']