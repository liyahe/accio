try:
    import torch
except:
    raise ModuleNotFoundError('torch must be installed')


# follow the sklearn implementation

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs.view(1, -1)
        v *= signs.view(-1, 1)
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs.view(1, -1)
        v *= signs.view(-1, 1)
    return u, v


class PCA_Torch(object):
    def __init__(self, n_components=2, whiten=False):
        self.n_components = n_components
        self.n_components_ = None
        self.whiten = whiten
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.n_samples_, self.n_features_ = None, None

    def fit(self, X):
        n_components = self.n_components
        n_samples, n_features = X.shape
        self.mean_ = torch.mean(X, dim=0)
        X_ = X - self.mean_

        # U, S, Vt = linalg.svd(X, full_matrices=False)
        U, S, Vt = torch.linalg.svd(X_, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        components_ = Vt

        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = torch.zeros_like(S).copy_(S)  # Store the singular values.

        if 0 < n_components < 1.0:
            ratio_cumsum = torch.cumsum(explained_variance_ratio_, dim=0)
            n_components = torch.searchsorted(ratio_cumsum, n_components, side='right') + 1

        self.n_components_ = n_components
        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = torch.mm(X_transformed, self.components_.T)
        if self.whiten:
            X_transformed /= torch.sqrt(self.explained_variance_)
        return X_transformed

#
# # test
# #
# import numpy as np
# import matplotlib.pyplot as plt
# 
#
# # Generate datasets
# def generate_data():
#     '''Generate 3 Gaussians samples with the same covariance matrix'''
#     n, dim = 512, 3
#     np.random.seed(0)
#     C = np.array([[1., 0.2, 0], [0.15, 1, 0.2], [0.1, 0.4, 10.0]])
#     X = np.r_[
#         np.dot(np.random.randn(n, dim), C),
#         np.dot(np.random.randn(n, dim), C) + np.array([1, 2, 5]),
#         np.dot(np.random.randn(n, dim), C) + np.array([-5, -2, 3]),
#     ]
#     y = np.hstack((
#         np.ones(n) * 0,
#         np.ones(n) * 1,
#         np.ones(n) * 2,
#     ))
#     return X, y
#
#
if __name__ == "__main__":
    from sklearn.decomposition import PCA

    X, y = generate_data()
    pca = PCA_Torch(n_components=0.1)
    X = torch.FloatTensor(X)
    pca.fit(X)
    trans_X = pca.transform(X)

    # fig = plt.figure(figsize=(8, 4))
    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(X.T[0], X.T[1], X.T[2], c=y)
    # ax = fig.add_subplot(122)
    # ax.scatter(trans_X.T[0], trans_X.T[1], c=y)
    # plt.show()

    X, y = generate_data()
    pca2 = PCA(n_components=0.1)
    pca2.fit(X)
    trans_X2 = pca2.transform(X)

    # fig = plt.figure(figsize=(8, 4))
    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(X.T[0], X.T[1], X.T[2], c=y)
    # ax = fig.add_subplot(122)
    # ax.scatter(trans_X2.T[0], trans_X2.T[1], c=y)
    # plt.show()
