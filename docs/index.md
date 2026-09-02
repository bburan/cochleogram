# Cochleogram

Cochleogram is a desktop application for analyzing confocal microscopy images of the cochlea. It supports two distinct workflows depending on how your tissue was imaged.

---

## Frequency map — full cochlea scan

Use this workflow when you imaged the entire cochlear length at **low magnification**, broken into multiple overlapping pieces from base to apex.

You align the image tiles, trace the IHC spiral across all pieces, and export a composite image with tonotopic frequency labels.

→ [Frequency Map workflow](frequency-map.md)

---

## Cell counting — ROI scan

Use this workflow when you imaged **specific frequency regions at high magnification** (e.g., 63×) to count hair cells.

You trace a spiral through each row of cells (IHC, OHC1–3), run automated detection, and correct the results manually. The spiral length is used to calculate cells per 100 µm.

→ [Cell Counting workflow](cell-counting.md)

---

## Supported file formats

| Format | Extra required |
|--------|---------------|
| Leica `.lif` | `pip install cochleogram[lif]` |
| Zeiss `.czi` | `pip install cochleogram[czi]` |
| Imaris `.ims` | `pip install cochleogram[ims]` |

## Supported species

| Species | Frequency range |
|---------|----------------|
| Mouse   | ~3–80 kHz      |
| Gerbil  | ~0.1–50 kHz    |

---

→ [Installation & setup](getting-started.md)
