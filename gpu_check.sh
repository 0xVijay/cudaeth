#!/bin/bash
# GPU Compatibility Check Script for CudaETH
# This script checks if your GPU is supported by the current build configuration

echo "=== CudaETH GPU Compatibility Check ==="
echo

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. NVIDIA drivers may not be installed."
    echo "Please install NVIDIA drivers and CUDA toolkit."
    exit 1
fi

# Get GPU information
echo "📊 Detected NVIDIA GPUs:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits | while IFS=, read -r name compute_cap; do
    name=$(echo "$name" | xargs)  # trim whitespace
    compute_cap=$(echo "$compute_cap" | xargs)
    
    echo "   GPU: $name"
    echo "   Compute Capability: $compute_cap"
    
    # Check if compute capability is supported
    case "$compute_cap" in
        5.0|5.2|5.3)
            echo "   Status: ✅ Supported (Maxwell architecture)"
            ;;
        6.0|6.1|6.2)
            echo "   Status: ✅ Supported (Pascal architecture - GTX 10 series)"
            ;;
        7.0|7.2)
            echo "   Status: ✅ Supported (Volta architecture - Tesla V100)"
            ;;
        7.5)
            echo "   Status: ✅ Supported (Turing architecture - RTX 20 series)"
            ;;
        8.0)
            echo "   Status: ✅ Supported (Ampere architecture - RTX 30 series/A100)"
            ;;
        8.6)
            echo "   Status: ✅ Supported (Ampere architecture - RTX 30 series)"
            ;;
        8.9)
            echo "   Status: ✅ Supported (Ada Lovelace - RTX 40 series)"
            ;;
        9.0)
            echo "   Status: ✅ Supported (Hopper/Lovelace Next - RTX 50 series)"
            ;;
        *)
            if [ $(echo "$compute_cap >= 5.0" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
                echo "   Status: ⚠️  May be supported (compute capability $compute_cap)"
            else
                echo "   Status: ❌ Not supported (compute capability $compute_cap < 5.0)"
            fi
            ;;
    esac
    echo
done

echo "=== Supported GPU Series ==="
echo "✅ GTX 10 Series (Pascal): GTX 1060, 1070, 1080, 1080 Ti"
echo "✅ RTX 20 Series (Turing): RTX 2060, 2070, 2080, 2080 Ti"  
echo "✅ RTX 30 Series (Ampere): RTX 3060, 3070, 3080, 3090, 3090 Ti"
echo "✅ RTX 40 Series (Ada Lovelace): RTX 4060, 4070, 4080, 4090"
echo "✅ RTX 50 Series (Future): RTX 5080, 5090 (when available)"
echo "✅ Tesla/Quadro: V100, A100, H100, RTX A6000, etc."
echo

echo "=== Build Instructions ==="
echo "To build CudaETH:"
echo "1. make clean && make"
echo "2. If you get architecture warnings, they can be safely ignored"
echo "3. The binary will support your GPU automatically"
echo