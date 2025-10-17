#!/bin/bash
#
# FynX Linting Script
# Runs code quality checks including formatting, import sorting, and type checking
#
# Usage:
#   ./scripts/lint.sh          # Check for issues
#   ./scripts/lint.sh --fix    # Automatically fix issues before checking
#

set -e

# Parse command line arguments
FIX_MODE=false
if [[ "$1" == "--fix" ]]; then
    FIX_MODE=true
    echo "🔧 Running FynX code quality checks with auto-fix..."
else
    echo "🔍 Running FynX code quality checks..."
fi
echo "======================================"

# Check if we're in a Poetry environment
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install Poetry and run 'poetry install --with dev'"
    exit 1
fi

# If in fix mode, automatically fix formatting and import issues
if [[ "$FIX_MODE" == true ]]; then
    echo "🔧 Auto-fixing code formatting and imports..."
    # Run isort first (sorts imports), then Black (formats everything)
    poetry run isort fynx tests examples
    poetry run black fynx tests examples
    echo "✅ Auto-fix completed"
    echo
    echo "🎉 Code has been automatically formatted and imports sorted!"
    echo "   (Skipping checks in fix mode - the fixes are applied)"
    echo
else
    # Run Black (code formatting)
    echo "📏 Checking code formatting with Black..."
    if ! poetry run black --check --diff fynx tests examples; then
        echo "❌ Black formatting check failed. Run './scripts/lint.sh --fix' to auto-fix."
        exit 1
    fi
    echo "✅ Black formatting check passed"

    # Run isort (import sorting)
    echo "🔀 Checking import sorting with isort..."
    if ! poetry run isort --check-only --diff fynx tests examples; then
        echo "❌ Import sorting check failed. Run './scripts/lint.sh --fix' to auto-fix."
        exit 1
    fi
    echo "✅ Import sorting check passed"
fi

# Run mypy (type checking) - match pre-commit hook configuration
echo "🔍 Running type checking with mypy..."
if ! poetry run mypy --ignore-missing-imports fynx tests examples; then
    echo "❌ Type checking failed. Fix the type errors before committing."
    exit 1
fi
echo "✅ Type checking passed"

echo ""
echo "🎉 All code quality checks passed!"
echo "==================================="
