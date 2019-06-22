import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm, orth

def _normal(X):
    """Normalize a matrix of observations in rows and variables in columns."""
    return (X - np.mean(X, axis=0)) @ np.linalg.inv(sqrtm(np.cov(X, rowvar=False)))


def _closed_form_rotation_solver(m):
    """Closed form rotation solver via SVD method."""
    u, _, vh = np.linalg.svd(m)
    return u @ vh


def _sinkhorn(p, q, X, Y, gamma, maxiter):
    """Entropy-Regularized (Sinkhorn) Optimal Transport between distributions"""
    x2, y2 = np.sum(np.power(X, 2), axis=0), np.sum(np.power(Y, 2), axis=0)
    C = np.tile(y2[np.newaxis, :], (X.shape[1], 1)) + np.tile(x2[:, np.newaxis], (1, Y.shape[1])) - 2 * (X.T @ Y)
    K = np.exp(-C / gamma)
    b = np.full(q.shape, 1 / len(q))
    for n in range(maxiter):
        a = p / (K @ b)
        b = q / (K.T @ a)
        if np.isnan(a).any():
            raise ArithmeticError('NaN found!')
    P = np.diag(a.squeeze()) @ K @ np.diag(b.squeeze())
    return P, np.sum(C * P)


def _sinkhorn_clusters(p, q, C, gamma, maxiter):
    """Entropy-Regularized (Sinkhorn) Optimal Transport between clusters"""
    K = np.exp(-C / gamma)
    b = np.ones(q.shape)
    for n in range(maxiter):
        a = p / (K @ b)
        b = q / (K.T @ a)

    P = np.diag(a.squeeze()) @ K @ np.diag(b.squeeze())

    return P, np.sum(C * P)


def _rMSE(X, Rg, Rgt):
    """Relative mean square error"""
    return np.linalg.norm(Rgt @ X.T - Rg @ X.T, 'fro') ** 2 / np.linalg.norm(Rgt @ X.T, 'fro') ** 2


def _eval_R2(X, Rg, Rgt):
    """Correlation coefficient"""
    X = (Rgt @ X.T).T
    Y = (Rg @ X.T).T
    return 1 - np.mean(Y - X, axis=0).sum() / np.var(Y, axis=0).sum()


class HiWA:
    """Hierarchical Wasserstein Alignment (HiWA)

    Applies nested OT between points XX and YY with decreasing entropy

    Parameters
    ----------
    dim_red_method : object with a fit_transform() method or None
        Method to compute a low-d embedding of the source and target distributions.
        Defaults to PCA.

    parallelize : Boolean
        Whether to use the multiprocessing module to run ADMM iterations concurrently.
        Defaults to True

    normal : Boolean
        Whether to normalize source and target before attempting alignment.
        Defaults to True, should be performed to prevent numerical errors.

    maxiter : int or None
        Maximum iterations for ADMM
        Defaults to 300

    tol : float, double, or None
        Stopping criterion for ADMM
        The change between rotation matrices on subsequent iterations must be
        greater than this (measured as Frobenius norm of the difference)
        Defaults to 0.1

    mu : float, double, or None
        ADMM parameter
        smaller = slower & more accurate, larger = faster & less accurate
        Defaults to 0.005

    shorn_maxiter : int or None
        Maximum iterations for Sinkhorn OT between clusters
        Defaults to 1000

    shorn_gamma : float, double, or None
        Entropy temperature for Sinkhorn OT between clusters
        larger = slower & more accurate, smaller = faster & less accurate
        Defaults to 0.2

    sa_maxiter : int or None
        Maximum iterations for subspace alignment procedure
        Defaults to 100

    sa_tol : float, double, or None
        Stopping criterion for subspace alignment procedure
        Defaults to 0.01

    sa_shorn_maxiter : int or None
        Maximum iterations for Sinkhorn OT within SA procedure
        Defaults to 150

    sa_shorn_gamma : float, double, or None
        Entropy temperature for Sinkhorn OT within SA procedure
        larger = slower & more accurate, smaller = faster & less accurate
        Defaults to 0.1
    """

    def __init__(self, dim_red_method=PCA(n_components=2), normal=True, maxiter=300,
                 tol=1e-1, mu=5e-3, shorn_maxiter=1000, shorn_gamma=2e-1, sa_maxiter=100,
                 sa_tol=1e-2, sa_shorn_maxiter=150, sa_shorn_gamma=1e-1):

        # Save parameters
        self.dim_red_method = dim_red_method
        self.normal = normal
        self.maxiter = maxiter
        self.tol = tol
        self.mu = mu
        self.shorn_maxiter = shorn_maxiter
        self.shorn_gamma = shorn_gamma
        self.sa_maxiter = sa_maxiter
        self.sa_tol = sa_tol
        self.sa_shorn_maxiter = sa_shorn_maxiter
        self.sa_shorn_gamma = sa_shorn_gamma

    def fit(self, X, X_labels, Y, Y_labels, **kwargs):
        """Fit the model with X, learning a rotation to match X to Y.

        Parameters
        ----------
        X : array-like, shape (n_samples_x, n_features)
            Source dataset, to be rotated.

        X_labels : array-like, shape (n_samples_x, )
            Cluster labels for the source dataset.

        Y : arraylike, shape (n_samples_y, n_features)
            Target dataset, of the same number of features as X, to rotate X to match.

        Y_labels : array-like, shape (n_samples_y, )
            Cluster labels for the target dataset.

        """
        # If not provided, compute transformations for source and target datasets using the method specified during
        # initialization
        X_transform = kwargs.get('X_transform', np.linalg.pinv(X) @ self.dim_red_method.fit_transform(X))
        Y_transform = kwargs.get('Y_transform', np.linalg.pinv(Y) @ self.dim_red_method.fit_transform(Y))
        self.Rgt = kwargs.get('Rgt', np.identity(X.shape[1]))
        if self.normal:
            X = _normal(X)
            Y = _normal(Y)

        # Initialization
        h_dim, num_clusters_x, num_clusters_y = X.shape[1], len(np.unique(X_labels)), len(np.unique(Y_labels))
        # Rg = np.identity(h_dim)
        Rg = _closed_form_rotation_solver(np.random.random((h_dim, h_dim)))
        P = np.full((num_clusters_x, num_clusters_y), 1 / (num_clusters_x * num_clusters_y))
        p = np.full((num_clusters_x, 1), 1 / num_clusters_x)
        q = np.full((num_clusters_y, 1), 1 / num_clusters_y)
        # Lagrangian multipliers
        L = np.zeros((h_dim, h_dim, num_clusters_x, num_clusters_y))
        # Auxiliary variables
        R = np.zeros((h_dim, h_dim, num_clusters_x, num_clusters_y))
        R[:, :, :, :] = np.identity(h_dim)[:, :, np.newaxis, np.newaxis]

        C = np.zeros((num_clusters_x, num_clusters_y))

        diagnostics = {'gamma': np.zeros(self.maxiter, dtype=float),
                       'Rg_norm': np.zeros(self.maxiter, dtype=float),
                       'rMSE': np.zeros(self.maxiter, dtype=float),
                       'R2': np.zeros(self.maxiter, dtype=float),
                       'C': np.zeros(C.shape, dtype=float)}

        # Compute low-d embeddings in high-d space and scale
        X_mbed = (X_transform @ X_transform.T @ X.T).T / np.sqrt(h_dim)
        Y_mbed = (Y_transform @ Y_transform.T @ Y.T).T / np.sqrt(h_dim)

        clust_ids_x = np.unique(X_labels)
        clust_ids_y = np.unique(Y_labels)

        # Distributed ADMM
        for n in range(self.maxiter):
            # Solve for each Q (potentially in parallel)

            for i in range(num_clusters_x):
                for j in range(num_clusters_y):
                    T = (self.mu / h_dim) * (Rg - L[:, :, i, j])
                    R[:, :, i, j], _, C[i, j] = self._subspace_alignment_solver(
                        X_mbed[(X_labels == clust_ids_x[i]).squeeze(), :],
                        Y_mbed[(Y_labels == clust_ids_y[j]).squeeze(), :],
                        P[i, j], T)

            # Solve for P
            P, _ = _sinkhorn_clusters(p, q, C, self.shorn_gamma, self.shorn_maxiter)

            # Solve for global rotation, Rg
            Rg_prev = Rg
            Rg = _closed_form_rotation_solver(
                np.mean(np.reshape(R + L, [h_dim, h_dim, num_clusters_x * num_clusters_y], order='F'), axis=2))

            # Update Lagrangian multipliers
            L = L + R - Rg[:, :, np.newaxis, np.newaxis]

            diagnostics['gamma'][n] = self.shorn_gamma
            diagnostics['Rg_norm'][n] = np.linalg.norm(Rg_prev - Rg, 'fro')
            diagnostics['rMSE'][n] = _rMSE(X, Rg, self.Rgt)
            diagnostics['R2'][n] = _eval_R2(X, Rg, self.Rgt)

            if (np.isnan(P).any() or diagnostics['Rg_norm'][n] <= self.tol) and n >= 5:
                diagnostics['gamma'] = diagnostics['gamma'][0:n + 1]
                diagnostics['Rg_norm'] = diagnostics['Rg_norm'][0:n + 1]
                diagnostics['rMSE'] = diagnostics['rMSE'][0:n + 1]
                diagnostics['R2'] = diagnostics['R2'][0:n + 1]
                diagnostics['C'] = C
                break

        self.Rg = Rg
        self.P = P
        self.diagnostics = diagnostics

    def transform(self, X):
        """Transform X by applying the learned rotation to it.

        Parameters
        ----------
        X : array-like, shape (n_samples_x, n_features)
            Source dataset, to be rotated.

        Returns
        ----------
        X_new : array-like, shape(n_samples_x, n_features)
        """
        return (self.Rg @ X.T).T

    def fit_transform(self, X, X_labels, Y, Y_labels, **kwargs):
        """Fit the model with X, learning a rotation to match X to Y, and apply the learned rotation to it.

        Parameters
        ----------
        X : array-like, shape (n_samples_x, n_features)
            Source dataset, to be rotated.

        X_labels : array-like, shape (n_samples_x, )
            Cluster labels for the source dataset.

        Y : arraylike, shape (n_samples_y, n_features)
            Target dataset, of the same number of features as X, to rotate X to match.

        Y_labels : array-like, shape (n_samples_y, )
            Cluster labels for the target dataset.

        Returns
        ----------
        X_new : array-like, shape(n_samples_x, n_features)

        """
        self.fit(X, X_labels, Y, Y_labels, **kwargs)
        return self.transform(X)

    def _subspace_alignment_solver(self, X, Y, P, T):
        """Earth Mover's Distance on low-rank projections"""
        # Initialization
        h_dim, num_x, num_y = X.shape[1], X.shape[0], Y.shape[0]
        R = orth(np.random.random((h_dim, h_dim)))
        # R = np.identity(h_dim)
        Q = np.full((num_x, num_y), 1 / (num_x * num_y))
        p = np.full((num_x, 1), 1 / num_x)
        q = np.full((num_y, 1), 1 / num_y)

        # Alternating minimization
        for i in range(self.sa_maxiter):
            R_prev = R

            # Solve rotation
            R = _closed_form_rotation_solver(2 * P * (Y.T @ Q.T @ X) + T)

            # Solve Sinkhorn OT
            Q, dist = _sinkhorn(p, q, R @ X.T, Y.T, self.sa_shorn_gamma / P, self.sa_shorn_maxiter)

            if np.linalg.norm(R_prev - R, 2) <= self.sa_tol:
                break

        return R, Q, dist
