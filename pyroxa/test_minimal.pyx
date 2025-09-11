# distutils: language = c++
# Simple test module to verify Cython compilation works with Python 3.13

def test_function():
    """Simple test function to verify module compilation"""
    return "Test successful"

cdef class TestClass:
    """Simple test class"""
    cdef public double value
    
    def __init__(self, double val=1.0):
        self.value = val
    
    def get_value(self):
        return self.value
