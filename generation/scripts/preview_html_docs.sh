#!/bin/bash

# FynX Documentation Preview Script
# Generates HTML documentation and starts the MkDocs development server
#
# Usage:
#   ./scripts/preview_docs.sh
#
# This will:
# 1. Generate HTML documentation from Markdown templates
# 2. Start a local development server at http://localhost:8000
# 3. Automatically reload when files change

set -e  # Exit on any error

echo "🚀 Starting FynX documentation preview..."

# Generate HTML documentation
echo "📝 Generating HTML documentation..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/generate_html.py"

# Start MkDocs development server
echo "🌐 Starting MkDocs development server..."
echo "   Documentation will be available at: http://localhost:8000"
echo "   Press Ctrl+C to stop the server"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MKDOCS_CONFIG="$SCRIPT_DIR/../mkdocs.yml"
poetry run mkdocs serve -f "$MKDOCS_CONFIG"
