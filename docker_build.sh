#!/bin/bash

# Docker build script for vastai/base-image environments
# This script builds the CUDA mnemonic enumeration tool with optimized settings
# to avoid compilation hanging/memory issues in resource-constrained containers

echo "=== Docker Build Script for CudaEth ==="
echo "Environment: vastai/base-image optimized"
echo

# Check if NVCC is available
if ! command -v nvcc &> /dev/null; then
    if [ -f /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        echo "✓ Added CUDA to PATH: /usr/local/cuda/bin"
    else
        echo "✗ NVCC not found. Please ensure CUDA is installed."
        exit 1
    fi
fi

echo "NVCC version:"
nvcc --version | head -4

echo
echo "=== Available Build Targets ==="
echo "1. minimal  - Minimal optimization (recommended for debugging)"
echo "2. docker   - Docker-optimized (balanced performance/memory)"  
echo "3. all      - Standard build (may cause memory issues)"
echo

# Default to minimal build
TARGET="minimal"
if [ "$1" != "" ]; then
    TARGET="$1"
fi

echo "Building with target: $TARGET"
echo

# Clean previous builds
echo "Cleaning previous builds..."
make clean

# Set memory limits if possible (helps prevent hanging)
if command -v ulimit &> /dev/null; then
    ulimit -v 4194304  # 4GB virtual memory limit
    echo "Set memory limit: 4GB"
fi

# Build with timeout to prevent hanging
echo "Starting compilation..."
echo "Note: If compilation hangs, it will timeout after 10 minutes"

timeout 600 make $TARGET

BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo
    echo "✓ Build successful!"
    echo "Executable location: build/bin/BruteForceMnemonic"
    
    if [ -f build/bin/BruteForceMnemonic ]; then
        echo "File size: $(ls -lah build/bin/BruteForceMnemonic | awk '{print $5}')"
    fi
elif [ $BUILD_EXIT_CODE -eq 124 ]; then
    echo
    echo "✗ Build timed out after 10 minutes"
    echo "This usually indicates a memory/resource issue."
    echo "Try running: ./docker_build.sh minimal"
    exit 124
else
    echo
    echo "✗ Build failed with exit code: $BUILD_EXIT_CODE"
    echo "Check the error messages above for details."
    exit $BUILD_EXIT_CODE
fi

echo
echo "=== Build Complete ==="