import numpy as np
try:
    import cupy as cp
except:
    pass
    


class MatrixInverse:

    def __init__(
        self,
        mode='numpy',
        pseudo_inv=False,
    ):

        self.mode = mode
        self.pseudo_inv = pseudo_inv
        if mode == 'cupy':
            self.inv = cp.linalg.pinv if self.pseudo_inv else cp.linalg.inv
        else:
            self.inv = np.linalg.pinv if self.pseudo_inv else np.linalg.inv

    def SchurComplement(self, X, full_matrix=True):

        if len(X.shape) < 2:
            raise Exception('The dimension of matrix should be 2, try .reshape(-1,1)')
        if X.shape[0] != X.shape[1]:
            raise Exception('The dimension of matrix should be square.')

        n = X.shape[0] // 2
        m = X.shape[0] - n

        E = X[:n, :n]
        F = X[:n, n:]
        G = X[n:, :n]
        H = X[n:, n:]

        Einv = self.inv(E)
        EF = - Einv @ F
        S = H + G @ EF
        D = self.inv(S)
        C = - D @ G @ Einv
        A = Einv - EF @ -C
        B = EF @ D

        Xinv = np.r_[np.c_[A, B], np.c_[C, D]]

        if full_matrix:
            return Xinv
        else:
            return A, B, C, D

    def fit(self, X):

        if self.mode == 'schur':
            self.Xinv = self.SchurComplement(X)
        else:
            self.Xinv = self.inv(X)