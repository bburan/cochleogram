# Getting Started

This guide covers installation and helps you choose the right workflow for your data.

---

## Installation

You need Python 3.7 or later. Install Cochleogram from PyPI using `pip`. Include the extras that match your file format:

```bash
# Leica (.lif) files
pip install cochleogram[lif]

# Zeiss (.czi) files
pip install cochleogram[czi]

# Both formats
pip install cochleogram[lif,czi]

# Imaris (.ims) files
pip install cochleogram[ims]
```

Verify the installation:

```bash
cochleogram --version
```

---

## Launching the app

```bash
cochleogram
```

To open a specific dataset immediately:

```bash
cochleogram /path/to/your/data
```

The app remembers the last dataset you opened.

---

## Which workflow do you need?

Cochleogram has two distinct modes. Choose based on how your tissue was imaged.

### Frequency map (full cochlea scan)

Use this when you imaged the entire cochlear length at **low magnification**, typically as a series of overlapping pieces from base to apex.

**Goal:** produce a composite image with tonotopic frequency labels.

**Steps:** align image tiles → trace the IHC spiral across all pieces → export the frequency map.

→ [Frequency Map workflow](frequency-map.md)

---

### Cell counting (ROI scan)

Use this when you imaged **specific frequency regions at high magnification** (e.g., 63×) to count hair cells.

**Goal:** count IHC and OHC cells per 100 µm at defined frequencies.

**Steps:** trace a spiral through each hair cell row → detect cells → review and correct.

→ [Cell Counting workflow](cell-counting.md)
