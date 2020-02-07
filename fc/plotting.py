"""
Methods for plotting simulation results.
"""
import operator
from functools import reduce

from .error_handling import ProtocolError

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.switch_backend('agg')  # on some machines this is required to avoid "Invalid DISPLAY variable" errors


def create_plot(path, x_data, y_data, x_label, y_label, title):
    """Creates 2d plots during protocols.

    :param path: A string path to write the plot to
    :param x_data: A list of arrays
    :param y_data: A list of arrays
    :param x_label: A label for the x axis
    :param y_label: A label for the y axis
    """

    # Check the x-axis data shape.
    # It must either be 1d, or be equivalent to a 1d vector (i.e. stacked copies of the same vector).
    for i, x in enumerate(x_data):
        if x.ndim > 1:
            num_repeats = reduce(operator.mul, x.shape[:-1])
            # Flatten all extra dimensions as an array view
            x_2d = x.reshape((num_repeats, x.shape[-1]))
            if x_2d.ptp(axis=0).any():
                # There was non-zero difference between the min & max at some position in the 1d equivalent vector
                raise ProtocolError(
                    'The X data for a plot must be (equivalent to) a 1d array, not', x.ndim, 'dimensions')
            x_data[i] = x_2d[0]  # Take just the first copy

    # Plot the data
    plt.figure()
    for i, x in enumerate(x_data):
        y = y_data[i]
        if y.ndim > 1:
            # Matplotlib can handle 2d data, but plots columns not rows, so we need to flatten & transpose
            y_2d = y.reshape((reduce(operator.mul, y.shape[:-1]), y.shape[-1]))
            plt.plot(x, y_2d.T)
        else:
            plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    plt.savefig(path)
    plt.close()
