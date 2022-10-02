import numpy as np
import unittest
from generators.stochastic_processes import Brownian, Lognormal, OU

SEED_VALUE = 5
MOCK_DATA = "tests/mock_data"


class TestStochasticProcesses(unittest.TestCase):

    def test_brownian(self):
        path = Brownian(delta_t=0.01)
        path.generate_path(seed=SEED_VALUE)
        t = np.load(f'{MOCK_DATA}/time.npy')
        y = np.load(f'{MOCK_DATA}/brownian.npy')
        self.assertTrue(all(t == path.t))
        self.assertTrue(all(y == path.y))

    def test_lognormal(self):
        path = Lognormal(delta_t=0.01)
        path.generate_path(seed=SEED_VALUE)
        t = np.load(f'{MOCK_DATA}/time.npy')
        y = np.load(f'{MOCK_DATA}/lognormal.npy')
        self.assertTrue(all(t == path.t))
        self.assertTrue(all(y == path.y))

    def test_ou(self):
        path = OU(delta_t=0.01)
        path.generate_path(seed=SEED_VALUE)
        t = np.load(f'{MOCK_DATA}/time.npy')
        y = np.load(f'{MOCK_DATA}/ou.npy')
        self.assertTrue(all(t == path.t))
        self.assertTrue(all(y == path.y))
