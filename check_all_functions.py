#!/usr/bin/env python3
"""Check if all functions are in __all__ list"""

import pyroxa

# Get all public functions
all_funcs = sorted([f for f in dir(pyroxa) if not f.startswith('_')])
print(f"Total public functions: {len(all_funcs)}")

# Get functions in __all__
all_list = sorted(pyroxa.__all__)
print(f"Functions in __all__: {len(all_list)}")

# Find missing functions
missing_from_all = [f for f in all_funcs if f not in all_list]
print(f"\nMissing from __all__: {len(missing_from_all)}")
for func in missing_from_all:
    print(f"  - {func}")

# Find extra functions in __all__
extra_in_all = [f for f in all_list if f not in all_funcs]
print(f"\nExtra in __all__ (not accessible): {len(extra_in_all)}")
for func in extra_in_all:
    print(f"  - {func}")

print(f"\n✓ Summary: {len(all_funcs)} total, {len(all_list)} in __all__, {len(missing_from_all)} missing")
