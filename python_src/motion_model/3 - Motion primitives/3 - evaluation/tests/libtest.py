
"""
libtest.py
==========

Unit test helpers

"""


def params(funcarglist):
    """Test function parameter decorator

    Parameters
    ----------

     * funcarglist: dict
    \tKeys and values become arguments and values in function

    """

    def wrapper(function):
        function.funcarglist = funcarglist
        return function
    return wrapper


def pytest_generate_tests(metafunc):
    """Enables params to work in py.test environment"""

    for funcargs in getattr(metafunc.function, 'funcarglist', ()):
        metafunc.addcall(funcargs=funcargs)
