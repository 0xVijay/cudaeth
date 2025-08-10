# Cross-platform Makefile for BruteForceMnemonic
# Usage: make (Linux/macOS) or make -f Makefile (Windows with MinGW)

# Compiler settings - Docker-compatible NVCC path detection
NVCC = $(shell [ -f /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc || echo nvcc)
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall
CUDAFLAGS = -O3 -Wno-deprecated-gpu-targets -diag-suppress 20044 \
  --generate-code arch=compute_50,code=sm_50 \
  --generate-code arch=compute_60,code=sm_60 \
  --generate-code arch=compute_61,code=sm_61 \
  --generate-code arch=compute_70,code=sm_70 \
  --generate-code arch=compute_75,code=sm_75 \
  --generate-code arch=compute_80,code=sm_80 \
  --generate-code arch=compute_86,code=sm_86 \
  --generate-code arch=compute_89,code=sm_89 \
  --generate-code arch=compute_90,code=sm_90

# Directories
SRCDIR = .
BUILDDIR = build
BINDIR = $(BUILDDIR)/bin
OBJDIR = $(BUILDDIR)/obj

# Source files
CUDA_SOURCES = $(wildcard BruteForceMnemonic/*.cu)
CXX_SOURCES = config/Config.cpp Tools/tools.cpp Tools/utils.cpp Tools/segwit_addr.cpp

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=$(OBJDIR)/%.o)
CXX_OBJECTS = $(CXX_SOURCES:%.cpp=$(OBJDIR)/%.o)

# Target executable
TARGET = $(BINDIR)/BruteForceMnemonic

# Default target
all: $(TARGET)

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

# Link everything together
$(TARGET): $(CUDA_OBJECTS) $(CXX_OBJECTS) | $(BUILDDIR)
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
	@echo "  all              - Build the project"
	@echo "  clean            - Remove build files"
	@echo "  install-deps-ubuntu  - Install dependencies on Ubuntu/Debian"
	@echo "  install-deps-centos  - Install dependencies on CentOS/RHEL"
	@echo "  help             - Show this help"

.PHONY: all clean install-deps-ubuntu install-deps-centos help
