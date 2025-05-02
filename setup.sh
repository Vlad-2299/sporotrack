#!/bin/bash
# Setup script for bite-o-serve
# This script installs the application and its dependencies using uv

set -e  # Exit on error

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing package mannager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv self update

echo "Creating virtual environment..."
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
uv pip install uvloop

echo "=================-Installation complete!-================="
echo "Run the program with the command: uv run main.py"
echo "====================================="