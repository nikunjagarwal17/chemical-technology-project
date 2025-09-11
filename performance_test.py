import time
import os

print("=== PyroXa Performance Test ===")

# Test pure Python
os.environ['PYROXA_PURE_PYTHON'] = '1'
import pyroxa

print(f"PyroXa version: {pyroxa.get_version()}")
print(f"Using pure Python mode")

# Time a computation-heavy function
print("\nTesting arrhenius_rate function...")
start = time.time()
for i in range(10000):
    result = pyroxa.arrhenius_rate(1e6, 50000, 298 + i*0.01)
python_time = time.time() - start

print(f'Pure Python time: {python_time:.3f}s')
print(f'Rate per second: {10000/python_time:.0f} calculations/sec')
print(f'Last result: {result:.2e}')

# Test another function
print("\nTesting autocatalytic_rate function...")
start = time.time()
for i in range(10000):
    result = pyroxa.autocatalytic_rate(0.1 + i*0.0001, 2.0, 1.5)
python_time2 = time.time() - start

print(f'Pure Python time: {python_time2:.3f}s')
print(f'Rate per second: {10000/python_time2:.0f} calculations/sec')
print(f'Last result: {result:.2e}')

print("\n=== Analysis ===")
if python_time < 1.0 and python_time2 < 1.0:
    print("✅ Pure Python is already very fast!")
    print("💡 C++ extensions may not be worth the complexity")
else:
    print("⚠️  Pure Python is slow for intensive calculations")
    print("💡 C++ extensions would provide significant benefits")
