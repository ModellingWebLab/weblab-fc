"""Test functionality of numpy arrays. No protocol arrays tested here."""

import numpy as np


def test_0d_array():
    array = np.array(0)
    assert array.ndim == 0  # number of dimensions is 0
    assert array.size == 1  # number of elements is 1
    assert len(array.shape) == 0  # size of shape is 0
    array[()] = 1  # note the different syntax for indexing a 0-d array
    assert array[()] == 1  # value of array is one


def test_basic_functionality():
    array = np.arange(4)  # assign array to equal consecutive integers
    assert array.ndim == 1  # number of dimensions is 1
    assert array.size == 4  # number of elements is 4
    assert array[1] == 1  # can reference from front
    assert array[-2] == 2  # can reference from back
    verify = 0
    for value in array:
        assert array[value] == verify
        verify += 1
    array2d = np.array([[0, 1], [0, 2], [1, 1]])  # can make 2-d array
    assert array2d.ndim == 2  # number of dimensions is 2
    assert len(array2d) == 3  # count number of subarrays in 2-d array
    assert array2d.size == 6  # count number of total elements in 2-d array
    array2dCopy = array2d.copy()  # can create true copy of array, altering array2dCopy does not alter array2d
    array2dCopy[2, 1] = 3
    assert array2d[2, 1] != array2dCopy[2, 1]
    array2dView = array2d  # creates view of array, altering array2dView alters array2d
    array2dView[2, 1] = 3
    assert array2d[2, 1] == array2dView[2, 1]
    array2d = array2d.reshape(1, 6)
    assert array2d.shape == (1, 6)  # you can reshape the arrays


def test_more_iteration_and_views():
    array = np.ones((3, 4, 2, 7))  # create four dimensional array of ones
    assert array.ndim == 4  # assert that there are four dimensions
    assert array.size == 3 * 4 * 2 * 7  # number of elements
    array = np.arange(168).reshape(3, 4, 2, 7)
    view = array[0::2, :0:-2, -1, 1::-1]  # can slice stepping forward, backward, picking a position...etc
    assert view.ndim == 3  # number of dimensions in this view is 3
    np.testing.assert_array_almost_equal(view, np.array([[[50, 49], [22, 21]], [[162, 161], [134, 133]]]))
    assert view.shape == (2, 2, 2)
    assert view.size == 2 * 2 * 2
    view[0, 0, 0] = 1  # change the first element of the view
    assert view[0, 0, 0] == array[0, 3, 1, 1]  # changing view changed the first element of the array
    copy = array.copy()
    assert copy[0, 0, 0, 0] == array[0, 0, 0, 0]  # copy is exact copy, first elements are equal
    copy[0, 0, 0, 0] = 10  # change first element of the copy
    assert copy[0, 0, 0, 0] != array[0, 0, 0, 0]  # changing the copy doesn't affect the original
    view = array[0, 0, 0, 1]
    assert view.ndim == 0  # you can take a 0-d view of an array and treat it as a value
    assert view.size == 1  # one element in the 0-d view
    assert view == 1  # the 0-d array is equal to the value of the number it contains

