import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from itertools import permutations
from mpl_toolkits.mplot3d import Axes3D

def load_data(file_name):
    mat_dict = loadmat(file_name, appendmat=True)
    return np.array(mat_dict['Tte']), np.array(mat_dict['Ttr']), np.array(mat_dict['Xte']), np.array(mat_dict['Xtr']), np.array(mat_dict['Yte']), np.array(mat_dict['Ytr'])


def normal(X):
    mean_X = np.mean(X, axis=0)
    cov_X = np.cov(X, rowvar=False)
    X_n = X
    #for col in range(X.shape[1]):
    #    X_n[:, col] = (X_n[:, col] - mean_X[col]) / np.sqrt(cov_X[col, col])
    return np.matmul(X_n - mean_X, np.linalg.inv(sqrtm(cov_X)))

def remove_const_cols(Y):
    return Y[:, ~np.all(Y[1:] == Y[:-1], axis=0)]

def map_X_3D(X):
    return np.column_stack((X[:, 0], X[:, 1], np.linalg.norm(X, axis=1)))

def LS_oracle(X_test, Y_test):
    X_n = X_test
    H_inv = X_n.T @ np.linalg.pinv(Y_test).T
    return H_inv

def plot_2d_clusters(X, labels):
    for i in np.unique(labels):
        plt.plot(X[labels == i, 0], X[labels == i, 1], linestyle='', marker='.', markersize=15)


def plot_3d_clusters(X, labels):
    for i in np.unique(labels):
        plt.scatter(X[id == i, 0], X[id == i, 1], X[id == i, 2], marker='.')


def eval_R2(X, Y):
    X = normal(X)
    Y = normal(Y)
    return 1 - np.mean(np.power(Y - X, 2), axis=0).sum() / np.var(Y, axis=0).sum()

def match_clustered_labels(X, X_labels, Y, Y_labels):
    """Find the optimal permutation of labels found by clustering, via 1-nearest neighbors classification accuracy.

    Parameters
    ----------
    X : array-like, shape (n_samples_x, n_features)
        Dataset of labels to match to.

    X_labels : array-like, shape (n_samples_x, )
        Labels to match to.

    Y : array-like, shape (n_samples_y, n_features)
        Dataset of labels to be matched.

    Y_labels : array-like, shape (n_samples_y, )
        Labels to be matched.

    Returns
    ----------
    Y_new : New Y labels.

    """
    nbrs = NearestNeighbors()
    nbrs.fit(X)
    idx = nbrs.kneighbors(Y, n_neighbors=1, return_distance=False).squeeze()

    Y_new = np.zeros(Y_labels.shape, dtype=int)
    for i in np.unique(Y_labels):
        idc = idx[Y_labels == i]
        Y_new[Y_labels == i], _ = mode(X_labels[idc])

    return Y_new
