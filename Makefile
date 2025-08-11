# Cross-platform Makefile for BruteForceMnemonic
# Usage: make (Linux/macOS/Unix)

# Compiler settings - Platform-independent NVCC path detection
NVCC = $(shell if [ -f /usr/local/cuda/bin/nvcc ]; then echo /usr/local/cuda/bin/nvcc; elif command -v nvcc >/dev/null 2>&1; then echo nvcc; else echo "echo 'NVCC not found'"; fi)
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -g0 -DNDEBUG
CUDAFLAGS = -O2 -Wno-deprecated-gpu-targets -diag-suppress 20044 -diag-suppress 191 -diag-suppress 177 \
  --maxrregcount=32 --use_fast_math \
  --generate-code arch=compute_70,code=sm_70 \
  --generate-code arch=compute_75,code=sm_75 \
  --generate-code arch=compute_80,code=sm_80 \
  --generate-code arch=compute_86,code=sm_86

# Directories
SRCDIR = .
BUILDDIR = build
BINDIR = $(BUILDDIR)/bin
OBJDIR = $(BUILDDIR)/obj

# Source files - CUDA-only build for Docker compatibility
CUDA_SOURCES = $(wildcard BruteForceMnemonic/*.cu)
# Temporarily disable C++ sources that have dependency issues
# CXX_SOURCES = config/Config.cpp Tools/tools.cpp Tools/utils.cpp Tools/segwit_addr.cpp

# Object files - CUDA-only for Docker
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=$(OBJDIR)/%.o)
# CXX_OBJECTS = $(CXX_SOURCES:%.cpp=$(OBJDIR)/%.o)

# Target executable
TARGET = $(BINDIR)/BruteForceMnemonic

# Default target
all: $(TARGET)

# Docker-optimized target (reduced memory usage)
docker: CUDAFLAGS = -O1 -Wno-deprecated-gpu-targets -diag-suppress 20044 -diag-suppress 191 -diag-suppress 177 \
  --maxrregcount=24 --use_fast_math -Xptxas=-v -Xptxas=-O1 --ptxas-options=-v \
  --generate-code arch=compute_75,code=sm_75
docker: CXXFLAGS = -std=c++17 -O1 -Wall -g0 -DNDEBUG
docker: $(TARGET)

# Minimal target for troubleshooting (single GPU arch, minimal optimization)
minimal: CUDAFLAGS = -O0 -Wno-deprecated-gpu-targets -diag-suppress 20044 -diag-suppress 191 -diag-suppress 177 \
  --maxrregcount=16 -Xptxas=-v --generate-code arch=compute_75,code=sm_75 \
  -Xcompiler -fno-strict-aliasing --compiler-options "-fno-stack-protector -fPIC"
minimal: CXXFLAGS = -std=c++17 -O0 -Wall -g0 -DNDEBUG
minimal: $(TARGET)

# Create directories
$(BUILDDIR):
	mkdir -p $(BUILDDIR)
	mkdir -p $(BINDIR)
	mkdir -p $(OBJDIR)/BruteForceMnemonic
	mkdir -p $(OBJDIR)/config
	mkdir -p $(OBJDIR)/Tools

# Compile CUDA files
$(OBJDIR)/%.o: %.cu | $(BUILDDIR)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@ -I$(SRCDIR) -I$(SRCDIR)/BruteForceMnemonic

# Compile C++ files
$(OBJDIR)/%.o: %.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(SRCDIR) -I$(SRCDIR)/BruteForceMnemonic

# Link everything together - CUDA-only build
$(TARGET): $(CUDA_OBJECTS) | $(BUILDDIR)
	$(NVCC) $(CUDAFLAGS) -o $@ $^ -lcudart

# Copy config and tables
	cp BruteForceMnemonic/config.cfg $(BUILDDIR)/
	mkdir -p $(BUILDDIR)/tables_ethereum
	cp tables_ethereum/A0.csv $(BUILDDIR)/tables_ethereum/

# Clean build files
clean:
	rm -rf $(BUILDDIR)

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	sudo apt-get update
	sudo apt-get install -y build-essential cmake nvidia-cuda-toolkit

# Install dependencies (CentOS/RHEL)
install-deps-centos:
	sudo yum groupinstall -y "Development Tools"
	sudo yum install -y cuda-toolkit

# Show help
help:
	@echo "Available targets:"
	@echo "  all              - Build the project (standard optimization)"
	@echo "  docker           - Build with Docker-optimized settings (reduced memory usage)"
	@echo "  minimal          - Build with minimal optimization for troubleshooting"
	@echo "  clean            - Remove build files"
	@echo "  install-deps-ubuntu  - Install dependencies on Ubuntu/Debian"
	@echo "  install-deps-centos  - Install dependencies on CentOS/RHEL"
	@echo "  help             - Show this help"

.PHONY: all clean docker minimal install-deps-ubuntu install-deps-centos help
