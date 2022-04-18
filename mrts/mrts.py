import numpy as np
from dissimilarity import Dissimilarity
from EigenDecomposition import EigenDecomposition
from MatrixInverse import MatrixInverse
from sklearn.preprocessing import StandardScaler

class MRTS:

    def __init__(
        self,
        if_scale = True,
        inv_mode = 'numpy',
        ed_mode = 'numpy',
        pseudo_inv = False,
        top_k=None,
        kernel_function='rbf',
        tolerance=1e-6,
    ):

        self.inv_mode = inv_mode
        self.ed_mode = ed_mode
        self.psuedo_inv = pseudo_inv
        self.return_top_k = top_k
        self.kernel_function = kernel_function
        self.if_scale = if_scale
        self.tolerance=tolerance

        self.MatrixInverse = MatrixInverse(mode=inv_mode, pseudo_inv=pseudo_inv)
        self.EigenDecomposition = EigenDecomposition(mode=ed_mode, sort=True, top_k=top_k)
        self.Dissimilarity = Dissimilarity(kernel_function=kernel_function, tolerance=tolerance)
    
    def fit(self, control_points, location):

        n = control_points.shape[0]
        dim = control_points.shape[1]

        X = np.c_[np.ones(shape=(n,1)), control_points]

        self.MatrixInverse.fit(X.T @ X)
        XTXinv = self.MatrixInverse.Xinv
        H = X @ XTXinv @ X.T
        Q = np.eye(n) - H
        self.Dissimilarity.fit(control_points, location)
        P = self.Dissimilarity.dist_mtx
        #P = radius_basis_function(dissimilarity(control_points, location), dim)
        self.EigenDecomposition.fit(Q @ P @ Q)
        vals, vecs = self.EigenDecomposition.vals, self.EigenDecomposition.vecs
        #vals, vecs = np.linalg.eig(Q @ P @ Q)
        vals, vecs = vals[:(n - dim - 1)], vecs[:,:(n - dim - 1)]

        basis = np.ones(shape=(n, n))
        basis[:,1:(dim + 1)] = control_points
        basis[:,(dim + 1):] = (P - P.T @ H).T @ vecs @ np.diag(1 / vals)

        if self.if_scale:
            sd = StandardScaler()
            basis = sd.fit_transform(basis)

        self.basis = basis