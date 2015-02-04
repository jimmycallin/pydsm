import numpy as np
import scipy.sparse as sp

def _dict2matrix(dmatrix):
    cdef float val
    cdef int rowindex

    if len(dmatrix) == 0:
        return sp.coo_matrix(np.ndarray((0,0))), [], []
   
    # Giving indices to words
    row2word= list(dmatrix.keys())
    col2word = list(set.union(*[set(col.keys()) for col in dmatrix.values()]))
    word2row = {w: i for i, w in enumerate(row2word)}
    word2col = {w: i for i, w in enumerate(col2word)}

    # Store as sparse coo matrix
    rows = []
    cols = []
    data = []
    for row, rowdict in dmatrix.items():
        rowindex = word2row[row]
        for col, val in rowdict.items():
            rows.append(rowindex)
            cols.append(word2col[col])
            data.append(val)

    return (sp.coo_matrix((data, (rows, cols)), shape=(len(row2word), len(col2word))),
            row2word,
            col2word)