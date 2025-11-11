#!/bin/bash

# =============================================================================
# Environment Setup Script for Image Similarity API
# =============================================================================
# This script helps you set up the correct environment configuration

echo "=========================================="
echo "Image Similarity API - Environment Setup"
echo "=========================================="
echo ""

# Function to show usage
show_usage() {
    echo "Usage: ./setup_env.sh [environment]"
    echo ""
    echo "Available environments:"
    echo "  dev         - Development environment"
    echo "  prod        - Production environment"
    echo "  ubuntu      - Ubuntu-specific configuration"
    echo "  windows     - Windows-specific configuration"
    echo ""
    echo "Example:"
    echo "  ./setup_env.sh dev"
    echo "  ./setup_env.sh ubuntu"
    echo ""
}

# Check if argument is provided
if [ -z "$1" ]; then
    echo "Error: No environment specified"
    echo ""
    show_usage
    exit 1
fi

ENV_TYPE=$1

# Determine source file
case $ENV_TYPE in
    dev|development)
        SOURCE_FILE=".env.development"
        ;;
    prod|production)
        SOURCE_FILE=".env.production"
        ;;
    ubuntu)
        SOURCE_FILE=".env.ubuntu"
        ;;
    windows)
        SOURCE_FILE=".env.windows"
        ;;
    *)
        echo "Error: Invalid environment type: $ENV_TYPE"
        echo ""
        show_usage
        exit 1
        ;;
esac

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Configuration file $SOURCE_FILE not found"
    echo ""
    echo "Please make sure you have the following files:"
    echo "  - .env.development"
    echo "  - .env.production"
    echo "  - .env.ubuntu"
    echo "  - .env.windows"
    echo ""
    echo "You can copy from .env.example and modify as needed."
    exit 1
fi

# Backup existing .env if it exists
if [ -f ".env" ]; then
    echo "Backing up existing .env to .env.backup..."
    cp .env .env.backup
fi

# Copy the selected environment file
echo "Setting up $ENV_TYPE environment..."
cp "$SOURCE_FILE" .env

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p models
mkdir -p index
mkdir -p shared/uploads

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Environment: $ENV_TYPE"
echo "Configuration file: $SOURCE_FILE -> .env"
echo ""
echo "Next steps:"
echo "1. Verify your .env file has correct values"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Setup the model: python setup_model.py"
echo "4. Run the API: uvicorn main:app --reload"
echo ""
echo "To test the setup, run: python main.py"
echo ""
