import numpy as np
import pathlib
import pyopencl as cl
from pyopencl import mem_flags as mf

kernel_file = pathlib.Path(__file__).parent.absolute()/"reduction.cl"
kernel_source = open(kernel_file).read()


def redsum(array, context, program):
    # Create queue
    queue = cl.CommandQueue(context)

    # Set the work-group size (number of work-items per work-group)
    group_size = 256

    # Pad an array
    a_h = pad(array, group_size)

    # Determine number of work-groups (work-items / work-group size)
    work_groups = a_h.shape[0]//group_size

    # Assign array of sum per work group
    p_h = np.zeros(work_groups)

    # Determine memory per work group (total size of array in bytes / number of
    # work groups, or, size of element of array in bytes * of work-group size)
    local_memory_size = a_h.nbytes//work_groups

    # Create buffers
    a_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_h)
    b_d = cl.LocalMemory(local_memory_size)
    p_d = cl.Buffer(context, mf.WRITE_ONLY, p_h.nbytes)

    # Call kernel
    redsum = program.sum
    redsum(queue, a_h.shape, (group_size,), a_d, b_d, p_d)
    cl.enqueue_copy(queue, p_h, p_d)

    # Sum of residuals
    return np.sum(p_h)


def redsum_axis0(array, context, program, group_size=256):
    # Create queue
    queue = cl.CommandQueue(context)

    # Pad array
    a_h = pad(array, group_size)
    print(a_h.shape)
    n_cols = a_h.shape[1]
    col_length = a_h.shape[0]

    # Determine number of work-groups (work-items / work-group size)
    work_groups_per_col = col_length//group_size

    # Assign array of sum per work group
    p_h = np.zeros((work_groups_per_col, n_cols))
    print(p_h.shape)

    # Determine memory per work group (total size of array in bytes / number of
    # work groups, or, size of element of array in bytes * of work-group size)
    local_memory_size = a_h[:, 0].nbytes//work_groups_per_col
    print(local_memory_size)

    # Create buffers
    a_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_h)
    b_d = cl.LocalMemory(local_memory_size)
    p_d = cl.Buffer(context, mf.WRITE_ONLY, p_h.nbytes)

    # Call kernel
    redsum = program.sum_axis0
    redsum(queue, a_h.shape, (group_size, 1,), a_d, b_d, p_d)
    cl.enqueue_copy(queue, p_h, p_d)

    # Sum residuals
    print(p_h)
    return np.sum(p_h, axis=0)


def redsum_axis1(array, context, program, group_size=256):
    # Create queue
    queue = cl.CommandQueue(context)

    # Pad array
    a_h = pad(array, group_size, axis=1)
    n_rows = a_h.shape[0]
    row_length = a_h.shape[1]

    # Determine number of work-groups (work-items / work-group size)
    work_groups_per_row = row_length//group_size

    # Assign array of sum per work group
    p_h = np.zeros((n_rows, work_groups_per_row))

    # Determine memory per work group (total size of array in bytes / number of
    # work groups, or, size of element of array in bytes * of work-group size)
    local_memory_size = a_h[0].nbytes//work_groups_per_row

    # Create buffers
    a_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_h)
    b_d = cl.LocalMemory(local_memory_size)
    p_d = cl.Buffer(context, mf.WRITE_ONLY, p_h.nbytes)

    # Call kernel
    redsum = program.sum_axis1
    redsum(queue, a_h.shape, (1, group_size,), a_d, b_d, p_d)
    cl.enqueue_copy(queue, p_h, p_d)

    # Sum residuals
    return np.sum(p_h, axis=1)


def pad(array, group_size, axis=0):
    """
    Pad an array with zeros so that it is a multiple of the group size.

    :arg array: Array to pad.
    :type array: :class:`numpy.ndarray`
    :arg int group_size: OpenCL group size.
    :arg int axis: The axis to pad with zeros. Default is 0.

    :returns: `array` padded with an appropriate number of zeros.
    :rtype: :class:`numpy.ndarray`
    """
    array_size = array.shape[axis]
    remainder = array_size % group_size
    if remainder == 0:
        return array
    else:
        padding = group_size - array_size % group_size
        padding_shape = list(array.shape)
        padding_shape[axis] = padding
        return np.concatenate(
            (array, np.zeros(padding_shape, dtype=array.dtype)), axis=axis
            )
