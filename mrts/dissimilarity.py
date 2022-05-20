import numpy as np

class Dissimilarity():

    def __init__(
        self,
        kernel_function = None,
        tolerance = 1e-6,
    ):

        self.kernel_function = kernel_function
        self.tolerance = tolerance

    def fit(self, X, Y=None):

        if Y is not None:
            dist_mtx = self.dissimilarity(X, Y)
        else:
            dist_mtx = self.dissimilarity(X, X)

        if self.kernel_function == 'rbf':
            dist_mtx = self.radius_basis_function(dist_mtx, dim=X.shape[1])
        elif self.kernel_function == None:
            dist_mtx = dist_mtx
        elif callable(self.kernel_function):
            dist_mtx = self.kernel_function(dist_mtx)

        self.dist_mtx = dist_mtx

    def radius_basis_function(self, dist, dim):

        r = dist + self.tolerance

        if dim == 1:
            return r ** 3 / 12
        elif dim == 2:
            return r ** 2 * np.log(r) / 8 / np.pi
        elif dim == 3:
            return - r / 8

    def dissimilarity(self, X, Y):

        X = X[None, :, :]
        Y = Y[:, None, :]
        diff = X - Y
        dist = np.linalg.norm(diff, ord=2, axis=2)

        return dist
