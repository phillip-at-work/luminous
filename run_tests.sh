#!/bin/bash

# Test runner script for luminous ray tracer

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "ERROR: pytest is not installed"
    echo "Install with: pip install pytest pytest-cov"
    exit 1
fi

# Run tests based on argument
case "${1:-}" in
    --debug|-d)
        # Debug mode: show print statements even when tests pass
        echo "Running tests in DEBUG mode (shows all print output)..."
        pytest tests/ -v -s
        ;;
    --coverage|-c)
        echo "Running tests with coverage..."
        pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
        ;;
    *)
        # Normal mode: vanilla pytest behavior
        pytest tests/ -v
        ;;
esac

exit $?
