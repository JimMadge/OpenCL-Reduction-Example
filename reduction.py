import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf


def pad(array, array_size, group_size):
    """Pad an array with zeros so that it is a multiple of the group size."""
    remainder = array_size % group_size
    if remainder == 0:
        return array
    else:
        padding = group_size - array_size % group_size
        return np.concatenate((array, np.zeros(padding)))


if __name__ == "__main__":
    # Create a context and  queue
    platform = cl.get_platforms()
    devices = platform[0].get_devices(cl.device_type.GPU)
    if devices == []:
        devices = platform[0].get_devices(cl.device_type.DEFAULT)
    context = cl.Context([devices[0]])
    queue = cl.CommandQueue(context)

    # Build kernel
    kernel_source = open("./reduction.cl").read()
    program = cl.Program(context, kernel_source).build()

    # Set the work-group size (number of work-items per work-group)
    group_size = 256

    # Create and pad an array
    # Notice the initial array is not of dimensions 2^n, or a multiple of 2^n
    array_size = 1024*10 + 3
    a_h = np.random.random(array_size)
    a_h = pad(a_h, array_size, group_size)
    assert a_h.shape[0] % group_size == 0

    # Determine number of work-groups (work-items / work-group size)
    work_groups = a_h.shape[0]//group_size

    # Assign array of sum per work group
    p_h = np.zeros(work_groups)

    # Determine memory per work group (total size of array in bytes / number of
    # work groups, or, size of element of array in bytes * of work-group size)
    local_memory_size = a_h.nbytes//work_groups
    print(work_groups)
    print(local_memory_size)

    # Create buffers
    a_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_h)
    b_d = cl.LocalMemory(local_memory_size)
    p_d = cl.Buffer(context, mf.WRITE_ONLY, p_h.nbytes)

    redsum = program.sum
    redsum(queue, a_h.shape, (group_size,), a_d, b_d, p_d)

    cl.enqueue_copy(queue, p_h, p_d)
    print(np.sum(a_h))
    print(np.sum(p_h))
    assert np.isclose(np.sum(a_h), np.sum(p_h))
