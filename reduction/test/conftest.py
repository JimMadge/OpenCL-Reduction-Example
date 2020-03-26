from .. import kernel_source
import pyopencl as cl
import pytest


@pytest.fixture(scope="session")
def context():
    platform = cl.get_platforms()
    devices = platform[0].get_devices(cl.device_type.GPU)
    if devices == []:
        devices = platform[0].get_devices(cl.device_type.DEFAULT)
    context = cl.Context([devices[0]])
    return context


@pytest.fixture(scope="session")
def program(context):
    program = cl.Program(context, kernel_source).build()
    return program
