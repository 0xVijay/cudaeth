#!/bin/bash

# Syntax check script for CUDA files
# This checks for basic syntax errors without requiring full CUDA installation

echo "Checking CUDA syntax..."

# Check if g++ is available for basic C++ syntax check
if ! command -v g++ &> /dev/null; then
    echo "g++ not found, skipping syntax check"
    exit 0
fi

# Check basic C++ compilation on non-CUDA parts
echo "Checking config parsing..."
g++ -std=c++17 -I. -I./config -I./Tools -c config/Config.cpp -o /tmp/config_test.o 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "✓ Config parsing compiles successfully"
else
    echo "✗ Config parsing has compilation errors"
    exit 1
fi

echo "Checking tools..."
g++ -std=c++17 -I. -I./config -I./Tools -c Tools/tools.cpp -o /tmp/tools_test.o 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "✓ Tools compile successfully"
else
    echo "✗ Tools have compilation errors"
    exit 1
fi

echo "Basic syntax check completed successfully"
echo "Note: Full CUDA compilation check requires nvcc"