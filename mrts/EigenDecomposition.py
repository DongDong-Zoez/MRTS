import numpy as np
try:
    import cupy as cp
except:
    pass
    
class EigenDecomposition:

    def __init__(
        self,
        mode = 'numpy',
        sort = True,
        top_k = None,
    ):

        self.mode = mode
        self.sort = sort
        self.top_k = top_k

    def fit(self, X):

        if self.mode == 'numpy':
            vals, vecs = self.numpy_ed(X)
        elif self.mode == 'svd':
            vals, vecs = self.numpy_svd(X)
        elif self.mode == 'cupy':
            vals, vecs = self.cupy_ed(X)

        if self.sort:
            sort_idx = np.argsort(vals)[::-1]
            vals, vecs = vals[sort_idx], vecs[:,sort_idx]

        self.vals, self.vecs = vals, vecs

    def cupy_ed(self, X):
        vals, vecs = cp.linalg.eigh(X)
        return vals, vecs

    def numpy_ed(self, X):
        vals, vecs = np.linalg.eig(X)
        return vals, vecs

    def numpy_svd(self, X):

        # X = USV^H (the rows of vh are the eigenvector of X^TX)
        u, s, vh = np .linalg.svd(X)
        vals, vecs = s ** 2, vh.T

        return vals, vecs