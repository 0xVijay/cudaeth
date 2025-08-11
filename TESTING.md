# GPU Kernel Implementation Testing Guide

This document outlines how to test the completed allowlist-based GPU kernel implementation.

## Implementation Summary

The GPU kernel now properly implements:

1. **Allowlist-based enumeration** with mixed-radix conversion for the first 11 mnemonic positions
2. **Automatic checksum calculation** and validation for the 12th word
3. **Single-target address matching** with direct comparison instead of table lookups
4. **Robust error handling** with bounds checking and validation

## Key Fixes Applied

### GPU Kernel Improvements

- **Bounds checking**: Added validation for zero candidate counts to prevent division by zero
- **Index validation**: Added bounds checking for digit indices in mixed-radix conversion
- **Overflow protection**: Added check for thread indices exceeding total combination space
- **Address comparison**: Fixed endianness issues by comparing addresses byte-by-byte

### Host-Side Validation

- **Data integrity**: Enhanced Dispatcher validation to check allowlist data before GPU transfer
- **Array bounds**: Added validation for candidate counts and array bounds
- **Target validation**: Added validation for target address to ensure it's not all zeros

## Testing Steps

### 1. Build the Project

```bash
# Make sure you have CUDA toolkit installed
which nvcc

# Build the project
make clean && make all
```

### 2. Test Configuration Files

Use the provided test configurations:

#### Small Test (test_config.json)
```bash
cp test_config.json config.json
./build/bin/BruteForceMnemonic
```

#### Original Example (config.json)
```bash
# Use the original config.json with the larger constraint set
./build/bin/BruteForceMnemonic
```

### 3. Expected Behavior

#### Normal Operation
- Program should display "Allowlist mode enabled: X combinations across first 11 positions"
- If single target mode: "Single target mode enabled for address: 0x..."
- Should show "ALLOWLIST MODE: Skipping table loading, using single target comparison"
- GPU kernels should enumerate combinations efficiently

#### Error Handling
- Invalid positions should be caught during config parsing
- Missing positions should be reported
- Invalid words should be rejected
- Zero candidate counts should be handled gracefully

### 4. Debug Output

The implementation includes comprehensive error messages:

```
Error: Position X has no candidate words
Error: Position X has too many candidates (Y > 128)
Error: Target address is all zeros
cudaMemcpyToSymbol to dev_candidate_counts failed!
```

### 5. Performance Verification

The allowlist mode should:
- Skip table loading when single_target_mode is enabled
- Enumerate only valid combinations (not random generation)
- Handle invalid combinations by returning empty mnemonics
- Compare addresses directly instead of hash table lookups

## Implementation Details

### Mixed-Radix Enumeration
```cuda
for (int i = 0; i < 11; i++) {
    uint16_t count = dev_candidate_counts[i];
    if (count == 0) return; // Error handling
    
    uint32_t digit = linear_idx % count;
    linear_idx = linear_idx / count;
    indices[i] = dev_candidate_indices[i][digit];
}
```

### Address Comparison
```cuda
// Fixed byte-by-byte comparison
uint8_t* hash_bytes = (uint8_t*)hash;
for (int i = 0; i < 20; i++) {
    if (hash_bytes[i] != dev_target_address[i]) {
        match = false;
        break;
    }
}
```

### Error Recovery
```cuda
if (count == 0 || digit >= count || linear_idx > 0) {
    mnemonic_phrase[0] = 0; // Empty mnemonic for invalid cases
    return;
}
```

## Configuration Examples

### Valid Configuration
```json
{
  "wallet_type": "ethereum",
  "mnemonic_length": 12,
  "ethereum": {
    "target_address": "0x543Bd35F52147370C0deCBd440863bc2a002C5c5",
    "derivation_path": "m/44'/60'/0'/0/2"
  },
  "word_constraints": [
    {
      "position": 0,
      "words": ["abandon", "ability"]
    }
    // ... all positions 0-11 must be covered
  ]
}
```

### Error Cases to Test

1. **Missing position**: Remove a position from word_constraints
2. **Invalid word**: Use a word not in BIP-39 wordlist
3. **Invalid address**: Use malformed Ethereum address
4. **Too many candidates**: Add >128 words to a position

## Troubleshooting

### Common Issues

1. **CUDA not found**: Install NVIDIA CUDA Toolkit
2. **Compilation errors**: Check g++ and nvcc versions
3. **Runtime errors**: Check GPU memory and CUDA driver
4. **Invalid mnemonics**: Verify BIP-39 word list compatibility

### Debug Mode

For detailed debugging, modify the kernel to add printf statements:

```cuda
#define DEBUG_ALLOWLIST
#ifdef DEBUG_ALLOWLIST
printf("Thread %d: linear_idx=%llu, position=%d, count=%d, digit=%d, index=%d\n", 
       idx, linear_idx, i, count, digit, indices[i]);
#endif
```

This implementation addresses the original comment about "Missing GPU Kernel Implementation" and "Incomplete Integration" by providing a complete, robust allowlist enumeration system with proper error handling and validation.