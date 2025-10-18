#!/bin/bash

# FynX Documentation Collection Script
# Generates both markdown and HTML documentation, then deploys to GitHub Pages
#
# Usage:
#   ./scripts/collect_docs.sh

set -e  # Exit on any error

echo "📚 FynX Documentation Collection Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "fynx" ]; then
    print_error "Please run this script from the FynX project root directory"
    exit 1
fi

# Set PYTHONPATH to include the project root for mkdocstrings
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT"

# Step 1: Generate HTML Documentation with MkDocs
print_status "Step 1: Generating HTML documentation with MkDocs..."
if poetry run python docs/generation/scripts/generate_html.py; then
    print_success "HTML documentation generated successfully"
else
    print_error "Failed to generate HTML documentation"
    exit 1
fi

# Step 2: Verify documentation was built
print_status "Step 2: Verifying documentation build..."
if [ -d "site" ] && [ -f "site/index.html" ]; then
    print_success "Documentation build verified - site/ directory created"
else
    print_error "Documentation build verification failed - site/ directory not found"
    exit 1
fi

# Step 3: Deploy to GitHub Pages
print_status "Step 3: Deploying to GitHub Pages..."
if poetry run mkdocs gh-deploy -f docs/generation/mkdocs.yml --force; then
    print_success "Documentation deployed to GitHub Pages successfully!"
    echo ""
    echo "🎉 Documentation collection complete!"
    echo ""
    echo "View your documentation at:"
    echo "  https://off-by-some.github.io/fynx/"
    echo ""
    echo "Local preview available with:"
    echo "  poetry run mkdocs serve -f docs/generation/mkdocs.yml"
else
    print_error "Failed to deploy to GitHub Pages"
    echo ""
    print_warning "You can still deploy manually with:"
    echo "  poetry run mkdocs gh-deploy -f docs/generation/mkdocs.yml"
    exit 1
fi

echo ""
print_success "All documentation tasks completed successfully!"
echo "   - HTML documentation generated"
echo "   - Documentation deployed to GitHub Pages"
