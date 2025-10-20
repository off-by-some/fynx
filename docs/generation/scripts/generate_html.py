#!/usr/bin/env python3
"""
FynX HTML Documentation Generator
Generates HTML documentation using MkDocs with mkdocstrings.

Usage:
    python docs/generation/scripts/generate_html.py
"""

import os
import subprocess
import sys
from pathlib import Path


def generate_html_docs() -> None:
    """Generate HTML documentation using MkDocs with mkdocstrings"""

    print("🌐 Generating FynX HTML documentation using MkDocs...")

    try:
        # Check that MkDocs configuration exists
        mkdocs_yml_path = Path(__file__).parent.parent / "mkdocs.yml"
        if not mkdocs_yml_path.exists():
            print(f"❌ MkDocs configuration not found at {mkdocs_yml_path}")
            print("   Please ensure docs/generation/mkdocs.yml exists")
            sys.exit(1)

        # Ensure docs directory and index.md exist
        markdown_dir = Path(__file__).parent.parent / "markdown"
        index_md = markdown_dir / "index.md"
        if not index_md.exists():
            print(f"❌ Index file not found at {index_md}")
            print("   Please ensure docs/generation/markdown/index.md exists")
            sys.exit(1)

        # Build the HTML documentation
        print("🏗️  Building HTML documentation...")
        mkdocs_config_path = Path(__file__).parent.parent / "mkdocs.yml"

        # Set PYTHONPATH to include the project root so mkdocstrings can find the fynx module
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent.parent.parent
        env["PYTHONPATH"] = str(project_root)

        # Change to the generation directory where mkdocs.yml is located
        generation_dir = Path(__file__).parent.parent

        result = subprocess.run(
            ["poetry", "run", "mkdocs", "build"],
            capture_output=True,
            text=True,
            cwd=str(generation_dir),
            env=env,
        )

        if result.returncode != 0:
            print("❌ HTML documentation generation failed")
            print("Error output:", result.stderr)
            sys.exit(1)

        print("✅ HTML documentation generated successfully!")
        print("   Output directory: ./site/")
        print("   To preview locally: python -m mkdocs serve")
        print("   To deploy to GitHub Pages: python -m mkdocs gh-deploy")

    except Exception as e:
        print(f"❌ Error generating HTML documentation: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    generate_html_docs()


if __name__ == "__main__":
    main()
