#!/bin/bash

# =============================================================================
# Python Environment Setup Script for Image Similarity API
# Ubuntu 22.04 - Miniconda Environment
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Image Similarity API - Python Environment Setup"
echo "Ubuntu 22.04 with Miniconda"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_info "Working directory: $SCRIPT_DIR"
echo ""

# =============================================================================
# Check Prerequisites
# =============================================================================
print_info "Checking prerequisites..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Miniconda not found. Please install Miniconda first."
    exit 1
fi
print_success "Miniconda found: $(conda --version)"

# Check Python version available
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "System Python version: $PYTHON_VERSION"
echo ""

# =============================================================================
# Conda Environment Setup
# =============================================================================
ENV_NAME="image-similarity-api"
PYTHON_VER="3.10"

print_info "Setting up conda environment: $ENV_NAME (Python $PYTHON_VER)"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Conda environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
        print_success "Environment removed"
    else
        print_info "Using existing environment"
    fi
fi

# Create conda environment if not exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    print_info "Creating conda environment with Python $PYTHON_VER..."
    conda create -n $ENV_NAME python=$PYTHON_VER -y
    print_success "Conda environment created"
else
    print_success "Conda environment exists"
fi

echo ""

# =============================================================================
# Activate Environment and Install Dependencies
# =============================================================================
print_info "Installing Python dependencies..."

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
conda activate $ENV_NAME
print_success "Activated conda environment: $ENV_NAME"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip
print_success "pip upgraded"

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    print_info "Installing packages from requirements.txt..."
    pip install -r requirements.txt
    print_success "Dependencies installed from requirements.txt"
else
    print_error "requirements.txt not found!"
    exit 1
fi

echo ""

# =============================================================================
# System Dependencies (Optional - for OpenCV)
# =============================================================================
print_info "Checking system dependencies for OpenCV..."

# Check if libGL is available (needed for OpenCV on headless servers)
if ! ldconfig -p | grep -q libGL.so; then
    print_warning "libGL not found. Installing system dependencies for OpenCV..."
    echo "You may need to run: sudo apt-get install -y libgl1-mesa-glx libglib2.0-0"
    print_warning "Skipping system package installation (requires sudo)"
else
    print_success "OpenCV system dependencies OK"
fi

echo ""

# =============================================================================
# Verify Installation
# =============================================================================
print_info "Verifying installation..."

# Test critical imports
python << END
import sys
print("Python version:", sys.version)

# Test imports
try:
    import fastapi
    print("✓ FastAPI:", fastapi.__version__)
except ImportError as e:
    print("✗ FastAPI import failed:", e)
    sys.exit(1)

try:
    import uvicorn
    print("✓ Uvicorn installed")
except ImportError as e:
    print("✗ Uvicorn import failed:", e)
    sys.exit(1)

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
except ImportError as e:
    print("✗ PyTorch import failed:", e)
    sys.exit(1)

try:
    import torchvision
    print("✓ TorchVision:", torchvision.__version__)
except ImportError as e:
    print("✗ TorchVision import failed:", e)
    sys.exit(1)

try:
    import numpy
    print("✓ NumPy:", numpy.__version__)
except ImportError as e:
    print("✗ NumPy import failed:", e)
    sys.exit(1)

try:
    import PIL
    print("✓ Pillow:", PIL.__version__)
except ImportError as e:
    print("✗ Pillow import failed:", e)
    sys.exit(1)

try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except ImportError as e:
    print("✗ OpenCV import failed:", e)
    sys.exit(1)

try:
    import faiss
    print("✓ FAISS installed")
except ImportError as e:
    print("✗ FAISS import failed:", e)
    sys.exit(1)

try:
    import asyncpg
    print("✓ AsyncPG:", asyncpg.__version__)
except ImportError as e:
    print("✗ AsyncPG import failed:", e)
    sys.exit(1)

try:
    import anthropic
    print("✓ Anthropic:", anthropic.__version__)
except ImportError as e:
    print("✗ Anthropic import failed:", e)
    sys.exit(1)

try:
    import openai
    print("✓ OpenAI:", openai.__version__)
except ImportError as e:
    print("✗ OpenAI import failed:", e)
    sys.exit(1)

try:
    import voyageai
    print("✓ VoyageAI:", voyageai.__version__)
except ImportError as e:
    print("✗ VoyageAI import failed:", e)
    sys.exit(1)

print("\n✓ All critical dependencies verified!")
END

if [ $? -eq 0 ]; then
    print_success "Installation verification passed!"
else
    print_error "Installation verification failed!"
    exit 1
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
print_success "Python Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Conda environment: $ENV_NAME"
echo "Python version: $PYTHON_VER"
echo ""
echo "To activate this environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Run: ./setup_directories.sh"
echo "  2. Configure your .env file"
echo "  3. Download model: python setup_model.py"
echo "  4. Start API: uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo "=========================================="
