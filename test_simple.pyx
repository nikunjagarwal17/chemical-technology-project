# Simple test extension for Python 3.13 compatibility
def hello():
    """Simple test function"""
    return "Hello from Cython!"

cdef double add_numbers(double a, double b):
    """Simple C function"""
    return a + b

def test_add(double x, double y):
    """Python wrapper for C function"""
    return add_numbers(x, y)
