import numpy as np
from .. import redsum_axis1


def test_axis1(context, program):
    # Notice the initial array is not of dimensions 2^n, or a multiple of 2^n
    array = np.random.random((3000, 1024*2+32))

    expected_sum = np.sum(array, axis=1)
    actual_sum = redsum_axis1(array, context, program)

    assert np.allclose(expected_sum, actual_sum)
