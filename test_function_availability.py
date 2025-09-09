import pyroxa

print('Total available functions:', len(dir(pyroxa)))
print('Functions in __all__:', len(pyroxa.__all__))

# Check if key functions are available
test_funcs = ['arrhenius_rate', 'equilibrium_constant', 'gibbs_free_energy']
for func in test_funcs:
    if hasattr(pyroxa, func):
        print(f'✓ {func} available')
    else:
        print(f'❌ {func} missing')
        
print('\nSample of available functions:')        
available = [name for name in dir(pyroxa) if not name.startswith('_')]
print(available[:20])

print('\nFunctions in __all__:')
print(pyroxa.__all__)
