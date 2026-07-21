# Documentation Organization

This folder contains the organized documentation structure for FynX.

## Directory Structure

```
docs/
├── README.md                    # This file
├── generation/                  # Documentation generation files
│   ├── mkdocs.yml               # MkDocs configuration
│   ├── markdown/                # Documentation source files
│   │   ├── tutorial/            # Tutorial content
│   │   ├── reference/           # API reference documentation
│   │   └── mathematical/        # Mathematical foundations
│   └── scripts/                 # Documentation generation scripts
│       ├── generate_html.py     # HTML documentation generator
│       └── preview_html_docs.sh # Preview server launcher
├── images/                      # Universal images and assets
│   ├── banner.svg               # Main logo/banner for documentation
│   ├── icon_350x350.png         # Icon for favicons and logos
│   ├── quick-start.svg          # Quick-start illustration
│   └── code-examples.svg        # Code example illustration
└── specs/
    └── v1.0-roadmap.md          # Archived design draft
```

## Usage

### Generating HTML Documentation

```bash
python docs/generation/scripts/generate_html.py
```

### Previewing Documentation

```bash
bash docs/generation/scripts/preview_html_docs.sh
```

This will:
1. Build HTML documentation using MkDocs with mkdocstrings
2. Start a local development server at http://localhost:8000

### Adding New Pages

1. Create markdown files in the appropriate subdirectory (`tutorial/`, `reference/`, or `mathematical/`)
2. Add entries to the `nav` section in `docs/generation/mkdocs.yml`
