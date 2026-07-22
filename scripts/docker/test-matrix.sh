#!/usr/bin/env bash
#
# Run the FynX test suite across supported Python Docker images in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

DEFAULT_PYTHONS=("3.9" "3.10" "3.11" "3.12" "3.13")
PYTHONS=(${FYNX_PYTHONS:-${DEFAULT_PYTHONS[*]}})

LOG_DIR="$PROJECT_ROOT/.docker-test-logs"
COVERAGE_DIR="$PROJECT_ROOT/.coverage-reports"
mkdir -p "$LOG_DIR" "$COVERAGE_DIR"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required for the Python matrix. Use './scripts/dev test --local' for the current interpreter." >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    PYTEST_ARGS=(
        --cov=fynx
        --cov-fail-under=80
        --color=yes
        --durations=10
    )
else
    PYTEST_ARGS=("$@")
fi

run_for_python() {
    local python_version="$1"
    local tag_version="${python_version//./-}"
    local image="fynx-test:py-${tag_version}"
    local log_file="$LOG_DIR/python-${python_version}.log"
    local coverage_file="$COVERAGE_DIR/.coverage-${python_version}"
    local coverage_xml="$COVERAGE_DIR/coverage-${python_version}.xml"

    {
        echo "== Python ${python_version} =="
        echo "$ docker build --build-arg PYTHON_VERSION=${python_version} -f ${DOCKERFILE} -t ${image} ${PROJECT_ROOT}"
        docker build \
            --build-arg "PYTHON_VERSION=${python_version}" \
            -f "$DOCKERFILE" \
            -t "$image" \
            "$PROJECT_ROOT"

        echo "$ docker run --rm ${image} poetry run pytest ..."
        docker run --rm \
            -e "COVERAGE_FILE=/workspace/.coverage-reports/.coverage-${python_version}" \
            -v "$PROJECT_ROOT:/workspace" \
            -w /workspace \
            "$image" \
            poetry run pytest \
                "${PYTEST_ARGS[@]}" \
                "--cov-report=xml:/workspace/.coverage-reports/coverage-${python_version}.xml"

        test -f "$coverage_file" || true
        test -f "$coverage_xml"
    } >"$log_file" 2>&1
}

echo "Running Python matrix in parallel: ${PYTHONS[*]}"
echo "Logs: $LOG_DIR"
echo "Coverage XML: $COVERAGE_DIR"

pids=()
for python_version in "${PYTHONS[@]}"; do
    run_for_python "$python_version" &
    pids+=("$!")
done

failed=0
failed_versions=()
for index in "${!pids[@]}"; do
    python_version="${PYTHONS[$index]}"
    pid="${pids[$index]}"
    if wait "$pid"; then
        echo "[OK] Python ${python_version}"
    else
        echo "[FAIL] Python ${python_version}"
        failed=1
        failed_versions+=("$python_version")
    fi
done

if [ "$failed" -ne 0 ]; then
    echo ""
    echo "Failures:"
    for python_version in "${failed_versions[@]}"; do
        log_file="$LOG_DIR/python-${python_version}.log"
        if [ -f "$log_file" ]; then
            echo ""
            echo "--- Python ${python_version} log ---"
            cat "$log_file"
        fi
    done
    exit 1
fi

echo "All Python matrix jobs passed."
