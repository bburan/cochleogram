# Frequency Map Workflow

Use this workflow when you have imaged the full cochlear length at low magnification. The cochlea is typically captured as a series of overlapping pieces running from the hook (base, high frequency) to the apex (low frequency).

**Goal:** produce a composite image with tonotopic frequency labels overlaid on the IHC spiral.

---

## File naming

Name each piece file with a numbered suffix in order from **base to apex**:

```
B009-8L_piece_1.lif   ← hook / base (high frequency)
B009-8L_piece_2.lif
B009-8L_piece_3.lif
...
B009-8L_piece_7.lif   ← apex (low frequency)
```

Cochleogram uses the piece numbers to assemble the cochlea in the correct order.

!!! note
    If a piece is missing (e.g., lost tissue), it can be substituted from another animal. Add a `_copied_<note>` suffix to the filename — the composite figure will label it accordingly.

---

## Step 1: Load the data

1. Open the **File** menu.
2. Choose **Full cochlea scan**.
3. Navigate to your file or folder.

Cochleogram detects the format automatically (`.lif`, `.czi`, or `.ims`) and loads each piece into its own tab.

---

## Step 2: Align tiles

!!! note
    Skip this step if each piece is a single image with no overlap.

Each piece may contain multiple overlapping tiles that need to be registered before tracing.

Switch to **Tile mode** by pressing `T`.

**Automatic alignment (recommended):**

Click **Align tiles**. Cochleogram cross-correlates adjacent tiles using the MyosinVIIa channel.

**Manual adjustment:**

1. Press `N` / `P` to cycle through tiles within a piece.
2. Use the **arrow keys** to nudge the selected tile.
3. Toggle **Highlight selected** to show a red border around the active tile.

Repeat for each piece before moving on.

---

## Step 3: Trace the spiral

Switch to **Spiral mode** by pressing `S`.

For a frequency map, you only need to trace one row. **IHC is the standard choice**, but any row works — the frequency calculation depends on distance along the spiral, not which row you use.

Select the row with `I` (IHC), then for each piece:

1. **Right-click** to place nodes along the hair cell row. You need at least 4 nodes per piece.
2. Cochleogram fits a smooth B-spline curve through your nodes automatically.
3. Nodes can be placed in any order — the app determines the correct path.

**Setting the basal origin:**

The first node placed becomes the origin (basal end). To reassign the origin to a different node, **Ctrl+right-click** that node.

The origin matters — it anchors the frequency scale. Make sure it sits at the basal (high-frequency) end of the cochlea.

Trace the spiral through every piece from base to apex before generating the map.

---

## Step 4: Save your work

Press **Ctrl+S** or go to **File → Save analysis**.

Analysis state is saved as JSON files alongside your data:

```
B009-8L_piece_1_analysis.json
B009-8L_piece_2_analysis.json
...
```

---

## Step 5: Generate the frequency map

Go to **File → Generate composite (frequency map)** or press **Ctrl+F**.

Configure the export:

| Setting | Description |
|---------|-------------|
| **Species** | Mouse or Gerbil — determines the tonotopic frequency equation |
| **Frequency range** | Low and high bounds (kHz) for the labels |
| **Octave step** | How often frequency labels appear along the spiral |
| **Channels** | Which fluorescence channels to include in the composite |
| **File format** | PDF (recommended for publication), TIFF, or JPEG |

Click **Generate** to export.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `T` | Tile mode |
| `S` | Spiral mode |
| `I` | Select IHC row |
| `1` / `2` / `3` | Select OHC row 1 / 2 / 3 |
| `N` / `P` | Next / previous tile |
| `Ctrl+S` | Save analysis |
| `Ctrl+F` | Generate frequency map |
