import numpy as np
import pandas as pd
try:
    import cupy as cp
except:
    pass
from logger import SmoothValue
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
        use_gpu=False,
        txt_path='logger.txt'
    ):

        self.inv_mode = inv_mode
        self.ed_mode = ed_mode
        self.pseudo_inv = pseudo_inv
        self.return_top_k = top_k
        self.kernel_function = kernel_function
        self.if_scale = if_scale
        self.tolerance=tolerance
        self.use_gpu=use_gpu
        self.txt_path=txt_path



        if use_gpu:
            self.inv_mode = 'cupy'
            self.ed_mode = 'cupy'
            try:
                import cupy as cp
            except ImportError:
                raise ImportError('No module named \'cupy\', try pip install cupy')

        SmoothValue.setting(file_path=self.txt_path)

        self.MatrixInverse = MatrixInverse(mode=self.inv_mode, pseudo_inv=pseudo_inv)
        self.EigenDecomposition = EigenDecomposition(mode=self.ed_mode, sort=True, top_k=top_k)
        self.Dissimilarity = Dissimilarity(kernel_function=kernel_function, tolerance=tolerance)
    
    def fit(self, control_points, location):
        if self.use_gpu:
            self._fit_gpu(control_points, location)
        else: 
            self._fit_cpu(control_points, location)

        SmoothValue.durationTime()
        SmoothValue.info(
            inv_mode=self.inv_mode,
            ed_mode=self.ed_mode,
            pseudo_inv=self.pseudo_inv,
            return_top_k=self.return_top_k,
            kernel_function=self.kernel_function,
            if_scale = self.if_scale,
            tolerance=self.tolerance,
            use_gpu=self.use_gpu,
            txt_path=self.txt_path,
            )
        SmoothValue.writeInfo()
        
    
    def _fit_cpu(self, control_points, location):

        n = control_points.shape[0]
        dim = control_points.shape[1]

        X = np.c_[np.ones(shape=(n,1)), control_points]

        self.MatrixInverse.fit(X.T @ X)
        SmoothValue.addCallback('Matrix Inverse')
        XTXinv = self.MatrixInverse.Xinv
        H = X @ XTXinv @ X.T
        Q = np.eye(n) - H
        self.Dissimilarity.fit(control_points, location)
        SmoothValue.addCallback('Dissimilarity')
        P = self.Dissimilarity.dist_mtx
        
        self.EigenDecomposition.fit(Q @ P @ Q)
        SmoothValue.addCallback('EigenDecomposition')
        vals, vecs = self.EigenDecomposition.vals, self.EigenDecomposition.vecs

        vals, vecs = vals[:(n - dim - 1)], vecs[:,:(n - dim - 1)]

        basis = np.ones(shape=(n, n))
        basis[:,1:(dim + 1)] = control_points
        basis[:,(dim + 1):] = (P - P.T @ H).T @ vecs @ np.diag(1 / vals)

        SmoothValue.addCallback('Build basis')
        
        if self.if_scale:
            sd = StandardScaler()
            basis = sd.fit_transform(basis)

        SmoothValue.addCallback('Normalize')

        self.basis = basis

    def _fit_gpu(self, control_points, location):

        control_points = cp.asarray(control_points)
        location = cp.asarray(control_points)

        n = control_points.shape[0]
        dim = control_points.shape[1]

        X = cp.c_[cp.ones(shape=(n,1)), control_points]

        self.MatrixInverse.fit(X.T @ X)
        SmoothValue.addCallback('Matrix Inverse')

        XTXinv = self.MatrixInverse.Xinv
        H = X @ XTXinv @ X.T
        Q = cp.eye(n) - H
        self.Dissimilarity.fit(control_points, location)
        SmoothValue.addCallback('Dissimilarity')
        del location
        P = self.Dissimilarity.dist_mtx
        
        self.EigenDecomposition.fit(Q @ P @ Q)
        SmoothValue.addCallback('Eigen Decomposition')
        del Q
        vals, vecs = self.EigenDecomposition.vals, self.EigenDecomposition.vecs

        vals, vecs = vals[:(n - dim - 1)], vecs[:,:(n - dim - 1)]

        basis = cp.ones(shape=(n, n))
        basis[:,1:(dim + 1)] = control_points
        basis[:,(dim + 1):] = (P - P.T @ H).T @ vecs @ cp.diag(1 / vals)
        del vecs, vals

        SmoothValue.addCallback('Build basis')

        if self.if_scale:
            mean = basis.mean(axis=0)
            std = basis.std(axis=0)
            basis = (basis - mean) / std

        SmoothValue.addCallback('Normalize')

        basis = basis.get()

        self.basis = basis
    
    @staticmethod
    def to_csv(basis, path='basis.csv'):
        pd.DataFrame(basis).to_csv(path)