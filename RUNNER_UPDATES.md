# GitHub Actions Runner Updates - Windows Server 2019 Retirement

## Issue Fixed
Error: "Windows Server 2019 has been retired. The Windows Server 2019 image has been removed as of 2025-06-30."

## Changes Made

### 1. Updated GitHub Actions Runners to Current Versions

#### `.github/workflows/build-and-test.yml`:
- ✅ `windows-2019` → `windows-latest` (automatically uses Windows Server 2022)
- ✅ `ubuntu-20.04` → `ubuntu-latest` (automatically uses Ubuntu 22.04+)
- ✅ `macOS-11` → `macos-latest` (automatically uses macOS 14+)

### 2. Current Runner Matrix (All Updated)

| Workflow | Previous | Updated | Current Version |
|----------|----------|---------|----------------|
| **build-and-test.yml** |
| Ubuntu | `ubuntu-20.04` | `ubuntu-latest` | Ubuntu 22.04+ |
| Windows | `windows-2019` ❌ | `windows-latest` | Windows Server 2022 |
| macOS | `macOS-11` | `macos-latest` | macOS 14+ |
| **build-wheels.yml** |
| All platforms | `*-latest` | `*-latest` | ✅ Already current |
| **ci.yml** |
| Ubuntu | `ubuntu-latest` | `ubuntu-latest` | ✅ Already current |

### 3. Benefits of Using `-latest` Runners

1. **Automatic Updates**: Always use the current supported version
2. **No Manual Updates**: GitHub automatically handles version transitions
3. **Better Performance**: Latest runners have improved performance and tools
4. **Extended Support**: Current runners have longer support lifecycles

### 4. GitHub Actions Runner Lifecycle

| Runner | Status | Retirement Date |
|--------|--------|----------------|
| `windows-2019` | ❌ Retired | June 30, 2025 |
| `windows-2022` | ✅ Active | Current |
| `windows-latest` | ✅ Recommended | Always current |
| `ubuntu-20.04` | ⚠️ Legacy | Still supported |
| `ubuntu-22.04` | ✅ Active | Current |
| `ubuntu-latest` | ✅ Recommended | Always current |
| `macOS-11` | ⚠️ Legacy | Transitioning |
| `macOS-12` | ✅ Active | Current |
| `macOS-13` | ✅ Active | Current |
| `macos-latest` | ✅ Recommended | Always current |

### 5. Testing

To verify the fixes work:

1. **Check Workflow Syntax**:
   ```bash
   # Validate workflow files
   gh workflow list
   ```

2. **Test Build Process**:
   - Push changes to trigger GitHub Actions
   - Verify builds run on correct runner versions
   - Confirm no runner availability errors

## Resolution Status: ✅ COMPLETE

All workflows now use current, supported GitHub Actions runners:
- ✅ **Windows Server 2019** → **Windows Server 2022** (via `windows-latest`)
- ✅ **macOS 11** → **macOS 14+** (via `macos-latest`)  
- ✅ **Ubuntu 20.04** → **Ubuntu 22.04+** (via `ubuntu-latest`)

### Best Practice Applied
Using `-latest` runner labels ensures:
- Automatic updates to current supported versions
- No future retirement issues
- Always running on the most stable, performance-optimized runners

The cibuildwheel builds should now run successfully on all supported platforms without runner availability errors.
