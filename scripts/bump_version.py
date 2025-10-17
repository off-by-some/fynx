#!/usr/bin/env python3
"""
Bump version script for FynX.

Updates the version number in both:
- pyproject.toml
- fynx/__init__.py

Usage:
    python scripts/bump_version.py <new_version>

Example:
    python scripts/bump_version.py 0.0.4
"""

import argparse
import re
import sys
from pathlib import Path


def update_pyproject_toml(new_version: str) -> None:
    """Update version in pyproject.toml"""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)

    content = pyproject_path.read_text()

    # Pattern to match version = "x.y.z" in pyproject.toml
    pattern = r'^version\s*=\s*".*?"$'
    replacement = f'version = "{new_version}"'

    if re.search(pattern, content, re.MULTILINE):
        updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        pyproject_path.write_text(updated_content)
        print(f"Updated version in {pyproject_path} to {new_version}")
    else:
        print(f"Error: Could not find version pattern in {pyproject_path}")
        sys.exit(1)


def update_init_py(new_version: str) -> None:
    """Update __version__ in fynx/__init__.py"""
    init_path = Path("fynx/__init__.py")

    if not init_path.exists():
        print(f"Error: {init_path} not found")
        sys.exit(1)

    content = init_path.read_text()

    # Pattern to match __version__ = "x.y.z"
    pattern = r'^__version__\s*=\s*".*?"$'
    replacement = f'__version__ = "{new_version}"'

    if re.search(pattern, content, re.MULTILINE):
        updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        init_path.write_text(updated_content)
        print(f"Updated __version__ in {init_path} to {new_version}")
    else:
        print(f"Error: Could not find __version__ pattern in {init_path}")
        sys.exit(1)


def validate_version_format(version: str) -> bool:
    """Validate version string format (x.y.z)"""
    pattern = r"^\d+\.\d+\.\d+$"
    return bool(re.match(pattern, version))


def main():
    parser = argparse.ArgumentParser(description="Bump version in FynX project")
    parser.add_argument("version", help="New version number (format: x.y.z)")
    args = parser.parse_args()

    new_version = args.version

    if not validate_version_format(new_version):
        print(f"Error: Invalid version format '{new_version}'. Expected format: x.y.z")
        sys.exit(1)

    try:
        update_pyproject_toml(new_version)
        update_init_py(new_version)
        print(f"\nVersion successfully bumped to {new_version}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
