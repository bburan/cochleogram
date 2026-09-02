"""
Capture screenshots of the Cochleogram app for documentation.

Usage:
    python scripts/capture_screenshots.py

Open Cochleogram with a dataset loaded, then run this script. It will guide
you through each screenshot: navigate the app to the right state, then press
Enter to capture. Screenshots are saved to docs/images/.
"""

import sys
from pathlib import Path

try:
    import pygetwindow as gw
    import pyautogui
except ImportError:
    print("Missing dependencies. Install them with:")
    print()
    print("    pip install pygetwindow pyautogui Pillow")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "images"

SHOTS = [
    (
        "tile-alignment.png",
        "TILE MODE — Press T to switch to Tile mode.\n"
        "  Make sure tiles are visible and overlapping.",
    ),
    (
        "spiral-tracing.png",
        "SPIRAL MODE — Press S to switch to Spiral mode.\n"
        "  Traces should be visible on the IHC and OHC rows.",
    ),
    (
        "cell-detection.png",
        "CELL MODE — Press C to switch to Cell mode.\n"
        "  Detected cells should be marked along the spirals.",
    ),
    (
        "frequency-map-dialog.png",
        "FREQUENCY MAP DIALOG — Press Ctrl+F to open the export dialog.\n"
        "  Leave it open before pressing Enter.",
    ),
]


def find_window():
    matches = [w for w in gw.getAllWindows() if "cochleogram" in w.title.lower()]
    if not matches:
        print("\nCochleogram window not found. Is the app open?")
        sys.exit(1)
    return matches[0]


def capture(win, dest):
    img = pyautogui.screenshot(region=(win.left, win.top, win.width, win.height))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)


def main():
    print("Cochleogram — documentation screenshot capture")
    print("=" * 48)

    win = find_window()
    print(f"\nFound: '{win.title}'")
    print(f"Saving to: {OUTPUT_DIR}\n")

    captured = []
    skipped = []

    for filename, instruction in SHOTS:
        print(f"\n{instruction}")
        try:
            input("  Press Enter to capture (Ctrl+C to skip this shot)... ")
        except KeyboardInterrupt:
            print(" skipped.")
            skipped.append(filename)
            continue

        win = find_window()
        dest = OUTPUT_DIR / filename
        capture(win, dest)
        print(f"  Saved → {dest.relative_to(Path(__file__).parent.parent)}")
        captured.append(filename)

    print("\n" + "=" * 48)
    print(f"Done. {len(captured)} captured, {len(skipped)} skipped.")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
