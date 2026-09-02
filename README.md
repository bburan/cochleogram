# Cochleogram

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://bburan.github.io/cochleogram/)

A GUI application for creating cochlear frequency maps and counting inner and outer hair cells from confocal microscopy images.

## What it does

Cochleogram loads confocal z-stack images of dissected cochleae, guides you through tracing hair cell rows, and produces:

- A **frequency map** — a composite image with tonotopic frequency labels along the cochlear spiral
- **Hair cell counts** — per-row tallies of Inner Hair Cells (IHC) and Outer Hair Cells (OHC1–3)

Supported file formats: Leica `.lif`, Zeiss `.czi`, Imaris `.ims`.

## Requirements

- Python 3.7 or later
- A supported confocal image file (`.lif`, `.czi`, or `.ims`)

## Installation

Install from PyPI. Include the extras for your file format:

```bash
# Leica (.lif) files
pip install cochleogram[lif]

# Zeiss (.czi) files
pip install cochleogram[czi]

# Both
pip install cochleogram[lif,czi]

# Imaris (.ims) files
pip install cochleogram[ims]
```

## Quick Start

Launch the application:

```bash
cochleogram
```

To open a dataset directly on launch:

```bash
cochleogram /path/to/your/data
```

Then follow the four-step workflow inside the app:

1. **Align tiles** — correct any overlap between z-stacks
2. **Trace spirals** — mark the IHC and OHC rows
3. **Detect cells** — auto-detect or manually mark individual hair cells
4. **Generate frequency map** — export the composite figure

## Documentation

Full documentation is available at the [GitHub Pages site](https://bburan.github.io/cochleogram/).

## License

See [LICENSE](LICENSE) for details.
