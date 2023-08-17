Cochleogram 
===========

Introduction
------------

This facilitates creating cochleograms from confocal images and will export
a frequency map and inner/outer hair cell positions along the tonotopic axis.
If position information is stored in the file, it can be used to automatically
align multiple confocal z-stacks for a single piece. 

Datasets
--------

Right now only Leica LIF files are supported. At one time we had support for
the Zeiss CZI format but it needs to be updated to work with the current
version of the program.

Naming conventions for LIF files
................................

Files should be named with the identifier (e.g., animal ID and ear) followed
byt the label for each channel in the order the channels were imaged. For
example:

    B009-8L-CtBP2-MyosinVIIa-PMT

Inside each file, the pieces must be numbered sequentially from base (hook) to
apex. If more htan one image is required for a piece, use letters for the
suffixes (i.e., "piece_2a", "piece_2b", etc.). The order in which the images
for a single piece are labeled does not matter since the program will
automatically align them based on the stage coordinates stored in the file.

Missing pieces
..............

If a piece is missing, you can copy the image stacks for a matching piece from
another file (for LIF files, you can use LAS X Office). To indicate that the
piece is a copy, it must have the suffix `copied_<note>`. For example, if you
copy piece_4a and piece_4b from B009-8L to the file containing data for
B021-3L, the copied images should be named "piece_4a_copy_B009-8L" and
"piece_4b_copy_B009-8L". The note will appear in the composite generated for
the frequency map.

Using the program
-----------------

Mouse interaction
.................
left click
    Select tile
left click + drag
    Pan image
mouse wheel
    Zoom in/out

Keyboard shortcuts
..................
t
    Switch to tile mode
i
    Switch to IHC mode
1
    Switch to OHC1 mode
2
    Switch to OHC2 mode
3
    Switch to OHC4 mode
4
    Switch to extra mode
s
    Select spiral tool
e
    Select exclude tool
c
    Select cell tool
n
    Select next tile (tile mode only)
p
    Select previous tile (tile mode only)
arrow keys
    The behavior of the arrow keys will depend on whether tile mode is
    selected. If tile mode is selected, then the arrow keys will move the tile.
    If any other mode is selected, the arrow keys will pan the image (this can
    be useful when in spiral or cell mode to move through the cochlea when
    zoomed in). To move the tile (or pan the image) in smaller steps, hold down
    shift at the same time.

Analysis
........

Analysis requires the following steps:

* Aligning the tiles so that they overlap as accurately as possible.
* Tracing a spiral through each row of hair cells.
* Marking individual hair cells.
* Marking regions containing uninterpretable data.

Tools are provided to facilitate each step. Be sure that you are satisfied with
the result of the current step before moving to the next step. Although you can
go back and edit a previous step, it may affect your analysis (e.g., if you
need to move a tile after marking hair cells, you may have to manually edit
a large number of hair cells).

**Tile mode**

Start by selecting "tiles" from the edit buttons, then left-clicking to select
the tile that is misaligned. Using the arrow keys, you can move the tile until
it is properly aligned with the other tiles. If you need to move the tile in
smaller steps, hold down the shift key at the same time as the arrow keys. It
may be helpful to toggle "highlight selected" so that you get a transparent
overlay. When in "highlight selected" mode, the currently selected tile will be
shown with a red border.

left click
    Select tile
mouse wheel
    Zoom in/out
arrow keys
    Move currently selected tile (large steps)
shift + arrow keys
    Move currently selected tile (small steps)
n
    Select next tile
p
    Select previous tile

An "align tiles" tool is provided to facilitate this step. It uses an automated
algorithm that attempts to align the tiles based on the correlation between the
images (using the MyosinVIIa channel).

**Spiral mode**

Once you are satisfied with the alignment of the tiles, select "IHC" from the
edit buttons and be sure the spiral tool to the right of the edit buttons are
selected. The very first point you mark should be on the end of the row of hair
cells facing the most basal region of the cochlea. This point will be
highlighted with a red circle. If you realize you made a mistake, you can
select a different point as the start of the spiral by control + right-clicking
that point when in spiral mode.

You must select a minimum of four points to create the spiral. You can add
points in between existing points and the spiral will be rerouted through those
points. The algorithm assumes that the "next" point in the path is the one
closest to it (i.e., the order in which you add the points does not
matter).

Repeat the process for OHC1, OHC2, and OHC3. Be sure that the spiral bisects
the nuclei (IHCs) or cuticular plate (OHCs) as that will facilitate the
semi-automated algorithms implemented by the program to help mark hair cells.

right click
    Add point
shift + right click
    Remove point
control + right click
    Set point as origin for spiral

**Cell mode**

After marking the spiral, run the algorithm to automatically detect cells. You
can play with the settings (each time you run, it will delete the existing
cells and create new ones). You will likely have to manually edit the
automatically-detected cells. Select the cell tool and then use right click to
add cells and shift + left click to delete cells.

right click
    Add cell
shift + right click
    Remove cell

From time to time there will be a fourth row of OHCs. These should manually be
identified by selecting "extra" for the cell you would like to edit and then
adding the cells using the cell tool.  Since the fourth row tends to be very
short in length, you cannot mark a spiral or mark the region as excluded.

**Exclude mode**

Finally, go back through each row of hair cells. If there was a region you felt
you could not intepret properly, select the exclude tool. Right-click the
spiral at one end of the region then right-click again at the other end of the
region you wish to exclude.

right click
    Start region. Click again to end region.
shift + right click
    Remove region under mouse cursor.
escape
    Cancel current region.

Some additional tools are made available to facilitate this process:

* You can merge all excluded regions across the OHC spirals into a single set
  of excluded regions that apply to all OHC spirals (`Combine OHC exclusions`
  button).
* You can simplify a set of excluded regions for a particular spiral if they
  are overlapping (this will combine overlapping exclusion regions into
  a single exclusion region) using the `Simplify exclusions` button.
