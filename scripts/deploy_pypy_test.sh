#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project root (one level up from scripts/)
cd "$SCRIPT_DIR/.."


# Verify we're in the right place
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Script must be in scripts/ subdirectory of project root."
    exit 1
fi

# Run linting and type checking
./scripts/lint.sh

# Run tests
poetry run pytest -q

# Build the package
poetry build

# Publish the package to the test PyPI repository
poetry publish -r testpypi
