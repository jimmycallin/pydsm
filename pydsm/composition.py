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




__dsm__ = ['compose']