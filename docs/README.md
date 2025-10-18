# Documentation Organization

This folder contains the organized documentation structure for FynX.

## Directory Structure

```
docs/
├── README.md                    # This file
├── index.md                     # Main landing page (MkDocs compatible)
├── generation/                  # Documentation generation files
│   ├── mkdocs.yml               # MkDocs configuration
│   ├── markdown/                # Generated markdown documentation
│   │   └── api.md               # API reference (processed by mkdocstrings)
│   └── scripts/                 # Documentation generation scripts
│       ├── generate_html.py     # HTML documentation generator
│       └── preview_docs.sh      # Preview server launcher
├── images/                      # Universal images and assets
│   ├── banner.svg               # Main logo/banner for documentation
│   ├── fynx_icon.svg            # Icon for favicons and logos
│   └── fynx.png                 # Legacy PNG image
└── specs/
    └── v1.0-roadmap.md          # v1.0 development roadmap
```

## Usage

### Generating HTML Documentation

```bash
python docs/generation/scripts/generate_html.py
```

### Previewing Documentation

```bash
bash docs/generation/scripts/preview_docs.sh
```

This will:
1. Build HTML documentation using MkDocs with mkdocstrings
2. Output to the `site/` directory

### Adding New Pages

1. Create markdown files directly in the `docs/` directory
2. Add entries to the `nav` section in `docs/build/mkdocs.yml`
