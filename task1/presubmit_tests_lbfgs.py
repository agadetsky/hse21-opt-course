import sys
from io import StringIO
import unittest
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_array_almost_equal, assert_allclose

from oracles import QuadraticOracle, create_log_reg_oracle
from optimization import lbfgs
from utils import LineSearchTool


# Check if it's Python 3
if not sys.version_info >= (3, 4):
    print('You should use python >= 3.4!')
    sys.exit()

test_bonus = 'bonus' in sys.argv


class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


def check_equal_histories(test_history, reference_history, atol=1e-3):
    if test_history is None or reference_history is None:
        assert_equal(test_history, reference_history)
        return

    for key in reference_history.keys():
        assert_equal(key in test_history, True)
        if key != 'time':
            assert_allclose(test_history[key], reference_history[key], atol=atol)
        else:
            # Cannot check time properly :(
            # At least, make sure its length is correct and its values are non-negative and monotonic
            assert_equal(len(test_history[key]), len(reference_history[key]))
            test_time = np.asarray(test_history['time'])
            assert_equal(np.all(test_time >= 0), True)
            assert_equal(np.all(test_time[1:] - test_time[:-1] >= 0), True)


class TestLBFGS(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    oracle = QuadraticOracle(A, b)

    f_star = -9.5
    x0 = np.array([0, 0])
    # For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_min, message, _ = lbfgs(self.oracle, self.x0)

        assert_equal(message, 'success')
        self.assertEqual(len(output), 0, 'You should not print anything by default.')

    def test_tolerance(self):
        """Check if argument `tolerance` is supported."""
        lbfgs(self.oracle, self.x0, tolerance=1e-5)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        lbfgs(self.oracle, self.x0, max_iter=15)

    def test_memory_size(self):
        """Check if argument `memory_size` is supported."""
        lbfgs(self.oracle, self.x0, memory_size=1)

    def test_line_search_options(self):
        """Check if argument `line_search_options` is supported."""
        lbfgs(self.oracle, self.x0, line_search_options={'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9})

    def test_display(self):
        """Check if something is printed when `display` is True."""
        with Capturing() as output:
            lbfgs(self.oracle, self.x0, display=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `display` is True.')

    def test_quality(self):
        x_min, message, _ = lbfgs(self.oracle, self.x0, tolerance=1e-5)
        f_min = self.oracle.func(x_min)

        g_k_norm_sqr = norm(self.A.dot(x_min) - self.b, 2)**2
        g_0_norm_sqr = norm(self.A.dot(self.x0) - self.b, 2)**2
        self.assertLessEqual(g_k_norm_sqr, 1e-5 * g_0_norm_sqr)
        self.assertLessEqual(abs(f_min - self.f_star), 1e-5 * g_0_norm_sqr)

    def test_history(self):
        x0 = -np.array([1.3, 2.7])
        x_min, message, history = lbfgs(self.oracle, x0,
                                        trace=True,
                                        memory_size=10,
                                        line_search_options={'method': 'Constant', 'c': 1.0},
                                        tolerance=1e-6)
        func_steps = [25.635000000000005,
                      22.99,
                      -9.3476294733722725,
                      -9.4641732176886055,
                      -9.5]
        grad_norm_steps = [11.629703349613008,
                           11.4,
                           0.55751193505619512,
                           0.26830541958992876,
                           0.0]
        time_steps = [0.0] * 5  # Dummy values
        x_steps = [np.array([-1.3, -2.7]),
                   np.array([1.0, 8.7]),
                   np.array([0.45349973, 3.05512941]),
                   np.array([0.73294321, 3.01292737]),
                   np.array([0.99999642, 2.99998814])]
        true_history = dict(grad_norm=grad_norm_steps, time=time_steps, x=x_steps, func=func_steps)
        check_equal_histories(history, true_history)

    @unittest.skipUnless(test_bonus, 'Skipping bonus test...')
    def test_history_best(self):
        x0 = -np.array([1.3, 2.7])
        x_min, message, history = lbfgs(self.oracle, x0,
                                        trace=True,
                                        memory_size=10,
                                        line_search_options={'method': 'Best'},
                                        tolerance=1e-6)
        func_steps = [25.635000000000005,
                      -8.8519395950378961,
                      -9.5]
        grad_norm_steps = [11.629703349613008,
                           1.1497712070693149,
                           0]
        time_steps = [0.0] * 3  # Dummy values
        x_steps = [np.array([-1.3, -2.7]),
                   np.array([-0.12706157, 3.11369481]),
                   np.array([1., 3.])]
        true_history = dict(grad_norm=grad_norm_steps, time=time_steps, x=x_steps, func=func_steps)
        check_equal_histories(history, true_history)


if __name__ == '__main__':
    argv = sys.argv
    if 'bonus' in argv:
        argv.remove('bonus')
    unittest.main(argv=argv)
