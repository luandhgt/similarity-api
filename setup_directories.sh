#!/bin/bash

# =============================================================================
# Directory Setup Script for Image Similarity API
# Creates all necessary directories including shared folders
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Image Similarity API - Directory Setup"
echo "Creating all necessary directories"
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
print_info "Parent directory: $(dirname "$SCRIPT_DIR")"
echo ""

# =============================================================================
# Create Project-Specific Directories
# =============================================================================
print_info "Creating project-specific directories..."

# Array of directories to create
PROJECT_DIRS=(
    "models"
    "index"
    "index/images"
    "index/name"
    "index/about"
    "logs"
    "config"
)

for dir in "${PROJECT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_warning "Directory already exists: $dir"
    else
        mkdir -p "$dir"
        print_success "Created: $dir"
    fi
done

echo ""

# =============================================================================
# Create Shared Directories (with parent project)
# =============================================================================
print_info "Creating shared directories..."

# Shared directory path (one level up from current project)
SHARED_BASE="$(dirname "$SCRIPT_DIR")/shared"

# Array of shared directories
SHARED_DIRS=(
    "$SHARED_BASE"
    "$SHARED_BASE/uploads"
    "$SHARED_BASE/uploads/events"
    "$SHARED_BASE/uploads/events/images"
    "$SHARED_BASE/uploads/temp"
)

for dir in "${SHARED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_warning "Shared directory already exists: $dir"
    else
        mkdir -p "$dir"
        print_success "Created shared: $dir"
    fi
done

echo ""

# =============================================================================
# Set Permissions (Optional)
# =============================================================================
print_info "Setting directory permissions..."

# Set write permissions for logs and uploads
chmod -R 755 logs 2>/dev/null || print_warning "Could not set permissions for logs"
chmod -R 755 index 2>/dev/null || print_warning "Could not set permissions for index"
chmod -R 755 "$SHARED_BASE/uploads" 2>/dev/null || print_warning "Could not set permissions for shared/uploads"

print_success "Permissions set"
echo ""

# =============================================================================
# Create .gitkeep files to preserve directory structure
# =============================================================================
print_info "Creating .gitkeep files..."

GITKEEP_DIRS=(
    "logs"
    "index/images"
    "index/name"
    "index/about"
    "$SHARED_BASE/uploads/events/images"
    "$SHARED_BASE/uploads/temp"
)

for dir in "${GITKEEP_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        touch "$dir/.gitkeep"
        print_success "Created .gitkeep in: $dir"
    fi
done

echo ""

# =============================================================================
# Verify Directory Structure
# =============================================================================
print_info "Verifying directory structure..."

# Check project directories
echo ""
echo "Project directories:"
for dir in "${PROJECT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "N/A")
        echo "  ✓ $dir ($SIZE)"
    else
        echo "  ✗ $dir (missing)"
    fi
done

# Check shared directories
echo ""
echo "Shared directories:"
for dir in "${SHARED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "N/A")
        # Show relative path for better readability
        REL_PATH=$(realpath --relative-to="$SCRIPT_DIR" "$dir")
        echo "  ✓ $REL_PATH ($SIZE)"
    else
        echo "  ✗ $dir (missing)"
    fi
done

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
print_success "Directory Setup Complete!"
echo "=========================================="
echo ""
echo "Created directories:"
echo "  - Project-specific: models, index, logs, config"
echo "  - Shared with Node.js: ../shared/uploads"
echo ""
echo "Directory tree structure:"
echo "  $(dirname "$SCRIPT_DIR")/"
echo "  ├── event-management/          (Node.js project)"
echo "  ├── image-similarity-api/      (Python project - current)"
echo "  │   ├── models/                (AI models)"
echo "  │   ├── index/                 (FAISS indices)"
echo "  │   │   ├── images/"
echo "  │   │   ├── name/"
echo "  │   │   └── about/"
echo "  │   ├── logs/                  (Application logs)"
echo "  │   └── config/                (Configuration files)"
echo "  └── shared/                    (Shared between projects)"
echo "      └── uploads/               (Uploaded files)"
echo "          ├── events/"
echo "          │   └── images/        (Event images)"
echo "          └── temp/              (Temporary files)"
echo ""
echo "Next steps:"
echo "  1. Configure .env file"
echo "  2. Download AI model: python setup_model.py"
echo "  3. Start the API server"
echo ""
echo "=========================================="
