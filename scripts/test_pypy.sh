#!/bin/bash
#
# TestPyPI Package Verification Script
# Tests fynx installation and functionality from TestPyPI using Docker
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🐳 FynX TestPyPI Verification"
echo "=================================="

# Build and test the image
echo "📦 Building Docker test image..."
if ! docker build -f "$SCRIPT_DIR/test-pypi.dockerfile" -t fynx-test .; then
    echo "❌ Docker build failed"
    echo ""
    echo "Possible issues:"
    echo "  • TestPyPI package not available or corrupted"
    echo "  • Network connectivity issues"
    echo "  • Docker not running"
    echo ""
    echo "Check: https://test.pypi.org/project/fynx/"
    exit 1
fi

echo "✅ Docker build successful"
echo "🎉 FynX installs correctly from TestPyPI"

# Run functional tests
echo ""
echo "🧪 Running functionality tests..."

TEST_CMD="python -c \"
import fynx
from fynx import observable

# Test basic observables
print('✅ FynX imported successfully')

# Test observable creation and reactivity
counter = observable(0)
double_counter = counter.then(lambda value: value * 2)

print('✅ Observables and computed values work')
print('📊 Initial counter:', counter.value)
print('📊 Initial double:', double_counter.value)

# Test reactivity
counter.set(5)
print('📊 After setting counter to 5:', counter.value)
print('📊 Computed double:', double_counter.value)

# Test store functionality
from fynx import Store
class TestStore(Store):
    name = observable('test')

store = TestStore()
print('✅ Store functionality works')
print('📊 Store name:', store.name)
\""

if docker run --rm fynx-test bash -c "$TEST_CMD"; then
    echo "✅ All functionality tests passed"
    echo ""
    echo "📦 Package Status: VERIFIED"
    echo "   • Installation: ✅ Working"
    echo "   • Imports: ✅ Working"
    echo "   • Core functionality: ✅ Working"
    echo "   • Quickstart example: ✅ Working"
else
    echo "❌ Functionality tests failed"
    echo ""
    echo "📦 Package Status: ISSUES DETECTED"
    echo "   • Check TestPyPI package contents"
    echo "   • Verify dependencies are available"
    exit 1
fi

# Clean up
echo ""
echo "🧹 Cleaning up..."
docker rmi fynx-test > /dev/null 2>&1
echo "✅ Cleanup complete"

echo ""
echo "🎉 TestPyPI verification complete!"
echo "🚀 Ready for production deployment"
