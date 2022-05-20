import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MatrixInverse import MatrixInverse
from dissimilarity import Dissimilarity

class TPS:

    def __init__(
      self, 
      kernel_function = 'rbf',
      tolerance = 1e-6,
      mode = 'numpy',
      target_type = None,
      pseudo_inv = True,
    ):
        self.tolerance = tolerance
        self.target_type = target_type
        self.custom_kernel = kernel_function
        self.pseudo_inv = pseudo_inv
        self.mode = mode

        self.Dissimilarity = Dissimilarity(kernel_function=kernel_function, tolerance=tolerance)
        self.MatrixInverse = MatrixInverse(mode=mode, pseudo_inv=pseudo_inv)

    def fit(self, control_points, deformation_points):

        if self.target_type == 'deformation':
            deformation_points = control_points + deformation_points

        if control_points.shape != deformation_points.shape:
            raise Exception('The shape of control_points and deformation points should be equal!')

        n = control_points.shape[0]
        d = control_points.shape[1]

        self.Dissimilarity.fit(control_points, control_points)
        K = self.Dissimilarity.dist_mtx

        #dist_mtx = self.dissimilarity(control_points, control_points)
        #K = self.radius_basis_function(dist_mtx, dim=d)
        P = np.c_[np.ones(shape=(n,1)), control_points]
        L_upper = np.c_[K, P]
        L_lower = np.c_[P.T, np.zeros(shape=(d+1, d+1))]
        L = np.r_[L_upper, L_lower]
        Y = np.r_[deformation_points, np.zeros(shape=(d+1,d))]

        if self.pseudo_inv:
            self.MatrixInverse.fit(L)
            coefs = self.MatrixInverse.Xinv.dot(Y)
        else:
            try:
                self.MatrixInverse.fit(L)
                coefs = self.MatrixInverse.Xinv.dot(Y)
            except:
                print('The matrix L is not invertible, change to pseudo inverse')
                self.MatrixInverse.fit(L)
                coefs = self.MatrixInverse.Xinv.dot(Y)

        self.d = d
        self.control_points = control_points
        self.Design_matrix = P
        self.Deformation_target = Y
        self.coefs = coefs[n:,:]
        self.weights = coefs[:n,:]

    def predict(self, newdata):

        design_mtx = np.c_[np.ones(shape=(newdata.shape[0],1)), newdata]

        self.Dissimilarity.fit(self.control_points, newdata)
        K = self.Dissimilarity.dist_mtx
        #dist_mtx = self.dissimilarity(self.control_points, newdata)

        plane = design_mtx.dot(self.coefs)
        smoothness = K.dot(self.weights)

        return plane + smoothness