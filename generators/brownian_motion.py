import matplotlib.pyplot as plt
import numpy as np


def brownian_path(mus=None,
                  sigmas=None,
                  T=1,
                  delta_t=10**(-4),
                  num_paths=1):
    if mus is None:
        mus = [0] * num_paths
    if sigmas is None:
        sigmas = [1] * num_paths
    if len(mus) != num_paths or len(sigmas) != num_paths:
        raise ValueError("The number of paths must match the number of Brownian motion parameters.")

    sqrt_delta_t = np.sqrt(delta_t)
    t = np.arange(0, T, delta_t)
    paths = []
    for _, mu, sigma in zip(range(num_paths), mus, sigmas):
        gaussians = [np.random.normal(mu, sigma * sqrt_delta_t) for _ in t]
        path = [sum(gaussians[:i+1]) for i in range(len(t))]
        paths.append(path)
    if len(paths) == 1:
        paths = paths[0]
    return t, paths

def correlated_brownian_paths(mus=None,
                              sigmas=None,
                              T=1,
                              delta_t=10**(-4),
                              num_paths=2,
                              corr_matrix=None):
    if mus is None:
        mus = [0] * num_paths
    if sigmas is None:
        sigmas = [1] * num_paths

    t, uncorrelated_paths = brownian_path(mus=mus, sigmas=sigmas, num_paths=num_paths)
    
    if corr_matrix is None:
        return t, uncorrelated_paths

    j = np.linalg.cholesky(corr_matrix)
    correlated_paths = np.array([j.dot(s) for s in np.array(uncorrelated_paths).T]).T
    return t, correlated_paths

def simple_corr_matrix(p, n):
    """
    Returns an nxn correlation matrix, s.t. all diagonal entries are 1 and all off-diagonal entries are p.
    """
    if p > 1 or p < -(1/(n-1)):
        raise ValueError(f"Invalid value of p: {p}. The allowed range is {-(1/(n-1))} <= p <= 1.")
    corr_matrix = np.ones((n, n))*p  + np.diag([(1-p)] * n)
    return corr_matrix


if __name__ == "__main__":
    num_paths = 5
    p = 0.99
    t, paths = brownian_path(num_paths=num_paths)
    corr_matrix = simple_corr_matrix(p, num_paths)
    t, correlated_paths = correlated_brownian_paths(num_paths=num_paths, corr_matrix=corr_matrix, delta_t=0.001)
    for path in paths:
        plt.plot(t, path, linestyle='--')
    plt.show()
    for path in correlated_paths:
        plt.plot(t, path, linestyle='--')
    plt.show()
