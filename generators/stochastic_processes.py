import matplotlib.pyplot as plt
import numpy as np


class Path:
    pass


class Brownian(Path):

    def __init__(self, mu=0, sigma=1, T=1, delta_t=10**(-4)):
        self.t = None
        self.y = None
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.delta_t = delta_t

    def generate_path(self, seed=None):
        np.random.seed(seed)
        sqrt_delta_t = np.sqrt(self.delta_t)
        self.t = np.arange(0, self.T, self.delta_t)
        x = 0
        self.y = [x]
        for _ in range(len(self.t)-1):
            x += self.mu * self.delta_t\
               + self.sigma * np.random.normal(0, sqrt_delta_t)
            self.y.append(x)


class BrownianJump(Path):

    def __init__(self, mu=0, sigma=1, T=1, delta_t=10**(-4), jump_intensity=1, jump_size=0):
        self.t = None
        self.y = None
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.delta_t = delta_t
        self.jump_intensity = jump_intensity
        self.jump_size = jump_size

    def generate_path(self, seed=None):
        np.random.seed(seed)
        sqrt_delta_t = np.sqrt(self.delta_t)
        self.t = np.arange(0, self.T, self.delta_t)
        x = 0
        self.y = [x]
        for _ in range(len(self.t)-1):
            x += self.mu * self.delta_t\
               + self.sigma * np.random.normal(0, sqrt_delta_t)\
               + self.jump(self.delta_t)
            self.y.append(x)

    def jump(self, time_interval):
        return self.jump_size * np.random.poisson(self.jump_intensity * time_interval, 1)[0]

    @classmethod
    def plot_example(self):
        num_paths = 20
        jump_size = -1
        jump_intensity = 0.5
        paths = [BrownianJump(jump_size=jump_size) for _ in range(num_paths)]
        for p in paths:
            p.generate_path()
        with plt.style.context('bmh'):
            for p in paths:
                plt.plot(p.t, p.y)
            plt.show()

class GeometricBrownian(Path):

    def __init__(self, mu=0, sigma=1, T=1, delta_t=10**(-4)):
        self.t = None
        self.y = None
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.delta_t = delta_t

    def generate_path(self, seed=None):
        np.random.seed(seed)
        sqrt_delta_t = np.sqrt(self.delta_t)
        self.t = np.arange(0, self.T, self.delta_t)
        x = 1
        self.y = [x]
        for _ in range(len(self.t)-1):
            x += x * self.mu * self.delta_t\
               + x * self.sigma * np.random.normal(0, sqrt_delta_t)
            self.y.append(x)


class OU(Path):
    """
    Ornstein-Uhlenbeck process.
    """

    def __init__(self, theta=0, sigma=1, T=1, delta_t=10**(-4)):
        self.t = None
        self.y = None
        self.theta = theta
        self.sigma = sigma
        self.T = T
        self.delta_t = delta_t

    def generate_path(self, seed=None):
        np.random.seed(seed)
        sqrt_delta_t = np.sqrt(self.delta_t)
        self.t = np.arange(0, self.T, self.delta_t)
        x = 0
        self.y = [x]
        for _ in range(len(self.t)-1):
            x += - self.theta * x * self.delta_t\
               + self.sigma * np.random.normal(0, sqrt_delta_t)
            self.y.append(x)


def correlate_brownian_paths_v2(paths, corr_matrix):
    ys = [p.y for p in paths]
    j = np.linalg.cholesky(corr_matrix)
    correlated_ys = np.array([j.dot(s) for s in np.array(ys).T]).T
    for p, y in zip(paths, correlated_ys):
        p.y = y
    return paths


def quadratic_variation(path, T=None):
    variation = 0
    for i, _ in enumerate(path.y[:-1]):
        variation += (path.y[i+1] - path.y[i])**2
    return variation


if __name__ == "__main__":
    BrownianJump.plot_example()
    num_paths = 20
    paths = [Brownian() for _ in range(num_paths)]
    for p in paths:
        p.generate_path()
    variations = [quadratic_variation(p) for p in paths]
    print(sum(variations)/len(variations))
    with plt.style.context('bmh'):
        for p in paths:
            plt.plot(p.t, p.y)
        plt.show()
