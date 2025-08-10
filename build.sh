#!/bin/bash

# Cross-platform build script for BruteForceMnemonic
# Usage: ./build.sh [clean|install-deps|build]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if CUDA is available
check_cuda() {
    if command -v nvcc &> /dev/null; then
        print_status "CUDA found: $(nvcc --version | head -n1)"
        return 0
    else
        print_error "CUDA not found. Please install NVIDIA CUDA Toolkit."
        return 1
    fi
}

# Check if required tools are available
check_dependencies() {
    local missing=()
    
    if ! command -v g++ &> /dev/null; then
        missing+=("g++")
    fi
    
    if ! command -v make &> /dev/null; then
        missing+=("make")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing[*]}"
        print_status "Run: ./build.sh install-deps"
        return 1
    fi
    
    print_status "All dependencies found"
    return 0
}

# Install dependencies based on OS
install_dependencies() {
    print_status "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            print_status "Installing dependencies on Ubuntu/Debian..."
            sudo apt-get update
            sudo apt-get install -y build-essential cmake
        elif command -v yum &> /dev/null; then
            print_status "Installing dependencies on CentOS/RHEL..."
            sudo yum groupinstall -y "Development Tools"
        else
            print_error "Unsupported Linux distribution"
            return 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "Installing dependencies on macOS..."
        if command -v brew &> /dev/null; then
            brew install cmake
        else
            print_error "Homebrew not found. Please install Homebrew first."
            return 1
        fi
    else
        print_error "Unsupported operating system: $OSTYPE"
        return 1
    fi
    
    print_status "Dependencies installed successfully"
}

# Clean build directory
clean_build() {
    print_status "Cleaning build directory..."
    rm -rf build/
    print_status "Build directory cleaned"
}

# Build the project
build_project() {
    print_status "Building BruteForceMnemonic..."
    
    if ! check_dependencies; then
        return 1
    fi
    
    if ! check_cuda; then
        return 1
    fi
    
    # Create build directory
    mkdir -p build/
    
    # Build using Makefile
    print_status "Compiling with Makefile..."
    make clean
    make all
    
    print_status "Build completed successfully!"
    print_status "Executable location: build/bin/BruteForceMnemonic"
    print_status "Configuration copied to: build/config.cfg"
    print_status "Tables copied to: build/tables_ethereum/"
}

# Main script logic
case "${1:-build}" in
    "clean")
        clean_build
        ;;
    "install-deps")
        install_dependencies
        ;;
    "build")
        build_project
        ;;
    "help")
        echo "Usage: $0 [clean|install-deps|build|help]"
        echo ""
        echo "Commands:"
        echo "  clean       - Clean build directory"
        echo "  install-deps - Install system dependencies"
        echo "  build       - Build the project (default)"
        echo "  help        - Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
