"""
Patch for Python 3.13 free-threaded library linking issue
"""

import sysconfig
import os

# Store original function
_original_get_config_var = sysconfig.get_config_var

def patched_get_config_var(name):
    """Patched version that handles free-threaded library naming"""
    value = _original_get_config_var(name)
    
    # If we're looking for library info and we have python313t.lib locally,
    # modify the configuration to use it
    if name == 'LIBRARY' and os.path.exists('python313t.lib'):
        return 'python313t.dll'  # Though this won't be used for linking
    elif name == 'LDLIBRARY' and os.path.exists('python313t.lib'):
        return 'python313t.dll'
    elif name == 'LIBDIR':
        # Add current directory to library search path
        if os.path.exists('python313t.lib'):
            return os.path.abspath('.')
    
    return value

# Apply the patch
sysconfig.get_config_var = patched_get_config_var

print("Applied Python 3.13 free-threaded library patch")
print(f"Library: {sysconfig.get_config_var('LIBRARY')}")
print(f"LDLIBRARY: {sysconfig.get_config_var('LDLIBRARY')}")
print(f"LIBDIR: {sysconfig.get_config_var('LIBDIR')}")
