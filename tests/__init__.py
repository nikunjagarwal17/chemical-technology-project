"""
PyroXa Test Suite
Pure Python implementation tests covering all 132+ functions
"""

# Test configuration
TEST_TOLERANCE = 1e-6
DEFAULT_TEMPERATURE = 298.15  # K
DEFAULT_PRESSURE = 101325    # Pa

# Common test utilities
def assert_close(actual, expected, tolerance=TEST_TOLERANCE):
    """Assert that two values are close within tolerance"""
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, got {actual}, difference: {abs(actual - expected)}")

def assert_positive(value):
    """Assert that a value is positive"""
    if value <= 0:
        raise AssertionError(f"Expected positive value, got {value}")

def assert_in_range(value, min_val, max_val):
    """Assert that a value is within a range"""
    if not min_val <= value <= max_val:
        raise AssertionError(f"Expected value in range [{min_val}, {max_val}], got {value}")