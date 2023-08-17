from copy import deepcopy

import json
from atom.api import (
    Atom,
    Bool,
    Dict,
    Enum,
    Event,
    Float,
    Int,
    List,
    observe,
    Property,
    Str,
    Tuple,
    Typed,
    Value,
)

from enaml.application import deferred_call

import matplotlib as mp
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib import (
    patheffects,
    ticker,
)
from matplotlib import patches as mpatches
from matplotlib import path as mpath
from matplotlib import transforms as T

import numpy as np
from scipy import interpolate

from cochleogram.model import ChannelConfig, Piece, Points, Tile
from cochleogram.util import get_region, make_plot_path, shortest_path


class PointPlot(Atom):

    #: Artist that plots the nodes the user specified
    artist = Value()

    #: Artist that draws the spline connecting the nodes the user selected. The
    #: spline is automatically updated as the user adds/removes nodes. The
    #: spline is used for many calculations so it is important to ensure it
    #: passes through the desired points.
    spline_artist = Value()

    axes = Value()
    points = Typed(Points)
    name = Str()
    visible = Bool(True)
    has_nodes = Bool(False)

    updated = Event()
    needs_redraw = Bool(False)

    def __init__(self, axes, points, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes
        self.artist, = axes.plot([], [], "ko", mec="w", mew=1, zorder=100)
        self.points = points
        points.observe('updated', self.request_redraw)

    def get_state(self):
        return {}

    def set_state(self, state):
        pass

    def add_point(self, x, y):
        self.points.add_node(x, y, hit_threshold=2.5)

    def set_origin(self, x, y):
        self.points.set_origin(x, y)

    def remove_point(self, x, y):
        self.points.remove_node(x, y)

    @observe("visible")
    def request_redraw(self, event=False):
        self.needs_redraw = True
        deferred_call(self.redraw_if_needed)

    def redraw_if_needed(self):
        if self.needs_redraw:
            self.redraw()
            self.needs_redraw = False

    def redraw(self, event=None):
        nodes = self.points.get_nodes()
        self.has_nodes = len(nodes[0]) > 0
        self.artist.set_data(*nodes)
        self.artist.set_visible(self.visible)
        self.updated = True


class LinePlot(PointPlot):

    spline_artist = Value()
    exclude_artist = Value()
    new_exclude_artist = Value()

    #: Artist that plots the starting point (i.e., the first node). This is a
    #: visual clue that lets the user know that the arc is proceeding in the
    #: right direction.
    origin_artist = Value()

    has_spline = Bool(False)
    has_exclusion = Bool(False)

    exclude_visible = Bool(False)

    start_drag = Value()
    end_drag = Value()

    def __init__(self, axes, points, **kwargs):
        super().__init__(axes, points, **kwargs)
        spline_effect = [
            patheffects.Stroke(linewidth=3, foreground="white"),
            patheffects.Normal(),
        ]
        (self.spline_artist,) = axes.plot(
            [], [], "k-", zorder=90, path_effects=spline_effect
        )
        self.origin_artist, = axes.plot([], [], "o", color='FireBrick', mec="w", mew=1, zorder=90, ms=10)

        verts = np.zeros((0, 2))
        path = mpath.Path(verts, [])
        self.new_exclude_artist = mpatches.PathPatch(path, facecolor='red', alpha=0.25, zorder=100)
        axes.add_patch(self.new_exclude_artist)

        verts = np.zeros((0, 2))
        path = mpath.Path(verts, [])
        self.exclude_artist = mpatches.PathPatch(path, facecolor='salmon', alpha=0.25, zorder=100)
        axes.add_patch(self.exclude_artist)

    def start_exclude(self, x, y):
        self.start_drag = x, y
        self.end_drag = None

    def update_exclude(self, x, y):
        self.end_drag = x, y
        self.request_redraw()

    def end_exclude(self, keep=True):
        if keep:
            self.points.add_exclude(self.start_drag, self.end_drag)
        self.start_drag = None
        self.end_drag = None
        self.request_redraw()

    def remove_exclude(self, x, y):
        self.points.remove_exclude(x, y)

    def _observe_exclude_visible(self, event=False):
        self.needs_redraw = True
        deferred_call(self.redraw_if_needed)

    def redraw(self, event=None):
        super().redraw()

        if self.has_nodes:
            nodes = self.points.get_nodes()
            self.origin_artist.set_data(nodes[0][0], nodes[1][0])
        else:
            self.origin_artist.set_data([], [])
        self.origin_artist.set_visible(self.visible)

        xi, yi = self.points.interpolate()
        self.has_spline = len(xi) > 0
        self.spline_artist.set_data(xi, yi)
        self.spline_artist.set_visible(self.visible)
        self.new_exclude_artist.set_visible(self.visible)

        self.has_exclusion = len(self.points.exclude) > 0
        path = make_plot_path(self.points, self.points.exclude)

        self.exclude_artist.set_path(path)
        self.exclude_artist.set_visible(self.exclude_visible)

        if self.start_drag and self.end_drag:
            try:
                regions = [(self.start_drag, self.end_drag)]
                path = make_plot_path(self.points, regions)
            except ValueError:
                # This usually means that region is too small to begin drawing.
                path = make_plot_path(self.points, [])
        else:
            path = make_plot_path(self.points, [])
        self.new_exclude_artist.set_path(path)


class ImagePlot(Atom):

    alpha = Float(0.75)
    highlight = Bool(False)
    zorder = Int(10)

    channel_config = Value()

    display_mode = Enum("projection", "slice")
    display_channels = List()
    extent = Tuple()
    z_slice = Int(0)
    z_slice_min = Int(0)
    z_slice_max = Int(0)
    shift = Float()

    tile = Typed(Tile)
    artist = Value()
    rectangle = Value()
    axes = Value()
    rotation_transform = Value()
    transform = Value()

    updated = Event()
    needs_redraw = Bool(False)

    def get_state(self):
        return {
            "alpha": self.alpha,
            "zorder": self.zorder,
            "display_mode": self.display_mode,
            "display_channels": self.display_channels,
            "z_slice": self.z_slice,
            "z_slice_min": self.z_slice_min,
            "z_slice_max": self.z_slice_max,
            "shift": self.shift,
        }

    def set_state(self, state):
        self.alpha = state["alpha"]
        self.zorder = state["zorder"]
        self.display_mode = state["display_mode"]
        self.display_channels = state["display_channels"]
        self.z_slice = state["z_slice"]
        self.z_slice_min = state["z_slice_min"]
        self.z_slice_max = state["z_slice_max"]
        self.shift = state["shift"]

    def __init__(self, axes, tile, **kwargs):
        super().__init__(**kwargs)
        self.tile = tile
        self.axes = axes
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        self.rotation_transform = T.Affine2D()
        self.transform = self.rotation_transform + axes.transData
        self.artist = axes.imshow(np.array([[0, 1], [0, 1]]), origin="lower", transform=self.transform)
        self.rectangle = mp.patches.Rectangle((0, 0), 0, 0, ec='red', fc='None', zorder=5000, transform=self.transform)
        self.rectangle.set_alpha(0)
        self.axes.add_patch(self.rectangle)
        self.z_slice_max = self.tile.image.shape[2] - 1
        self.z_slice = self.tile.image.shape[2] // 2
        self.shift = self.tile.info["voxel_size"][0] * 5
        self.channel_config = {c: ChannelConfig(name=c) for c in tile.channel_names}
        for config in self.channel_config.values():
            config.observe('visible', self.request_redraw)
            config.observe('min_value', self.request_redraw)
            config.observe('max_value', self.request_redraw)
        tile.observe('extent', self.request_redraw)

    def _observe_highlight(self, event):
        if self.highlight:
            self.rectangle.set_alpha(1)
        else:
            self.rectangle.set_alpha(0)

    def _observe_alpha(self, event):
        self.artist.set_alpha(self.alpha)

    def _observe_zorder(self, event):
        self.artist.set_zorder(self.zorder)

    def drag_image(self, dx, dy):
        extent = np.array(self.tile.extent)
        extent[0:2] += dx
        extent[2:4] += dy
        self.tile.extent = extent.tolist()

    def move_image(self, direction, step_scale=1):
        extent = np.array(self.tile.extent)
        step = step_scale * self.shift
        if direction == "up":
            extent[2:4] += step
        elif direction == "down":
            extent[2:4] -= step
        elif direction == "left":
            extent[0:2] -= step
        elif direction == "right":
            extent[0:2] += step
        self.tile.extent = extent.tolist()

    @observe("z_slice", "display_mode", "alpha", "highlight")
    def request_redraw(self, event=False):
        self.needs_redraw = True
        deferred_call(self.redraw_if_needed)

    def redraw_if_needed(self):
        if self.needs_redraw:
            self.redraw()
            self.needs_redraw = False

    def redraw(self, event=None):
        z_slice = None if self.display_mode == 'projection' else self.z_slice
        channels = [c for c in self.channel_config.values() if c.visible]
        image = self.tile.get_image(channels=channels, z_slice=z_slice).swapaxes(0, 1)
        self.artist.set_data(image)
        xlb, xub, ylb, yub = extent = self.tile.get_image_extent()[:4]
        self.artist.set_extent(extent)
        self.rectangle.set_bounds(xlb, ylb, xub-xlb, yub-ylb)
        t = self.tile.get_image_transform()
        self.rotation_transform.set_matrix(t.get_matrix())
        self.updated = True

    def contains(self, x, y):
        return self.tile.contains(x, y)

    def set_channel_visible(self, channel_name, visible):
        self.channel_config[channel_name].visible = visible

    def set_channel_min_value(self, channel_name, min_value):
        self.channel_config[channel_name].min_value = min_value

    def set_channel_max_value(self, channel_name, max_value):
        self.channel_config[channel_name].max_value = max_value


class Presenter(Atom):

    # Interface to help read data
    reader = Value()

    # Tile artists
    tile_artists = Dict()
    current_artist_index = Value()
    current_artist = Value()

    # For spirals and cells
    point_artists = Dict()
    current_spiral_artist = Value()
    current_cells_artist = Value()

    figure = Typed(Figure)
    axes = Typed(Axes)
    piece = Typed(Piece)

    highlight_selected = Bool(False)
    alpha_selected = Float(0.50)
    alpha_unselected = Float(0.50)
    zorder_selected = Int(20)
    zorder_unselected = Int(10)

    cells = Enum("IHC", "OHC1", "OHC2", "OHC3", "Extra")
    tool = Enum("tile", "spiral", "exclude", "cells")

    pan_event = Value()
    pan_xlim = Value()
    pan_ylim = Value()

    #: True if we actually had a pan event. This allows us to distinguish
    #: between clicks that select a tile vs. clicks that are intended to start
    #: a pan.
    pan_performed = Bool(False)

    drag_event = Value()
    drag_x = Value()
    drag_y = Value()

    spiral_empty = Bool(True)
    spiral_ready = Bool(False)
    cells_empty = Bool(True)

    unsaved_changes = Bool(False)
    needs_redraw = Bool(False)
    saved_state = Dict()

    z_min = Property()
    z_max = Property()

    def _get_z_min(self):
        return min(a.z_slice_min for a in self.tile_artists.values())

    def _get_z_max(self):
        return min(a.z_slice_max for a in self.tile_artists.values())

    def __init__(self, piece, reader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piece = piece
        self.reader = reader
        self.tile_artists = {
            t.source: ImagePlot(self.axes, t) for t in self.piece.tiles
        }
        for artist in self.tile_artists.values():
            artist.observe('updated', self.update)
        self.current_artist_index = 0
        for key in ('IHC', 'OHC1', 'OHC2', 'OHC3', 'Extra'):
            cells = PointPlot(self.axes, self.piece.cells[key], name=key)
            spiral = LinePlot(self.axes, self.piece.spirals[key], name=key)
            cells.observe('updated', self.update)
            spiral.observe('updated', self.update)
            self.point_artists[key, 'cells'] = cells
            self.point_artists[key, 'spiral'] = spiral

        # This is necessary because `imshow` will override some axis settings.
        # We need to set them back to what we want.
        self.axes.axis('equal')
        self.axes.axis(self.piece.get_image_extent())
        self.saved_state = self.get_full_state()
        self.drag_event = None

    def _observe_saved_state(self, event):
        self.check_for_changes()

    def check_for_changes(self):
        self.unsaved_changes = self.saved_state['data'] != \
            self.get_full_state()['data']

    def update(self, event=None):
        self.check_for_changes()
        self.needs_redraw = True
        deferred_call(self.redraw_if_needed)

    def _default_figure(self):
        return Figure()

    def _default_axes(self):
        return self.figure.add_axes([0, 0, 1, 1])

    def _observe_current_artist_index(self, event):
        self.current_artist = list(self.tile_artists.values())[self.current_artist_index]
        self.update_highlight()

    @observe("highlight_selected",)
    def update_highlight(self, event=None):
        alpha = self.alpha_unselected if self.highlight_selected else 1
        for artist in self.tile_artists.values():
            artist.zorder = self.zorder_unselected
            artist.alpha = alpha
            artist.highlight = False
        if self.current_artist is not None:
            if self.highlight_selected:
                self.current_artist.alpha = self.alpha_selected
                self.current_artist.rectangle.set_alpha(1)
                self.current_artist.highlight = True
            self.current_artist.zorder = self.zorder_selected
        self.redraw()

    @observe('cells', 'tool')
    def _update_plots(self, event=None):
        for artist in self.point_artists.values():
            artist.visible = False
            if hasattr(artist, 'exclude_visible'):
                artist.exclude_visible = False
        if self.tool == 'tile':
            self.current_spiral_artist = None
            self.current_cells_artist = None
        else:
            self.current_spiral_artist = self.point_artists[self.cells, 'spiral']
            self.current_cells_artist = self.point_artists[self.cells, 'cells']
            self.current_spiral_artist.exclude_visible = True
            if self.tool in ('spiral', 'exclude'):
                self.current_spiral_artist.visible = True
            else:
                self.current_cells_artist.visible = True

    def set_interaction_mode(self, cells=None, tool=None):
        if cells is not None:
            self.cells = cells
        if tool is not None:
            self.tool = tool

    def action_auto_align_tiles(self):
        self.piece.align_tiles()

    def action_guess_cells(self, width, spacing, channel):
        n = self.piece.guess_cells(self.cells, width, spacing, channel)
        self.set_interaction_mode(None, 'cells')
        return n

    def action_clear_cells(self):
        self.piece.clear_cells(self.cells)
        self.set_interaction_mode(None, 'cells')

    def action_clear_spiral(self):
        self.piece.clear_spiral(self.cells)
        self.set_interaction_mode(None, 'spiral')

    def action_clone_spiral(self, to_spiral, distance):
        xn, yn = self.piece.spirals[self.cells].expand_nodes(distance)
        self.piece.spirals[to_spiral].set_nodes(xn, yn)

    def action_copy_exclusion(self, to_spiral):
        if not self.point_artists[to_spiral, 'spiral'].has_spline:
            raise ValueError(f'Must create spiral for {to_spiral} first')
        for s, e in self.piece.spirals[self.cells].exclude:
            self.piece.spirals[to_spiral].add_exclude(s, e)

    def action_merge_exclusion(self, *spirals):
        exclude = []
        for spiral in spirals:
            if not self.point_artists[spiral, 'spiral'].has_spline:
                raise ValueError(f'Must create spiral for {to_spiral} first')
            exclude.extend(self.piece.spirals[spiral].exclude)
        for spiral in spirals:
            self.piece.spirals[spiral].exclude = exclude
            self.piece.spirals[spiral].simplify_exclude()

    def action_simplify_exclusion(self, *spirals):
        for spiral in spirals:
            self.piece.spirals[spiral].simplify_exclude()

    def key_press(self, event):
        key = event.key.lower()
        if key == 's':
            deferred_call(self.set_interaction_mode, None, 'spiral')
        elif key == 'e':
            deferred_call(self.set_interaction_mode, None, 'exclude')
        elif key == 'c':
            deferred_call(self.set_interaction_mode, None, 'cells')
        elif key == 't':
            deferred_call(self.set_interaction_mode, None, 'tile')
        elif key == 'i':
            deferred_call(self.set_interaction_mode, 'IHC', None)
        elif key == '1':
            deferred_call(self.set_interaction_mode, 'OHC1', None)
        elif key == '2':
            deferred_call(self.set_interaction_mode, 'OHC2', None)
        elif key == '3':
            deferred_call(self.set_interaction_mode, 'OHC3', None)
        elif key == '4':
            deferred_call(self.set_interaction_mode, 'Extra', None)
        elif (key == 'escape') and (self.drag_event is not None) and (self.tool == 'exclude'):
            self.end_drag_exclude(event, keep=False)
        elif self.tool == 'tile' and self.current_artist is not None:
            self.key_press_tile(event)
        else:
            self.key_press_point_plot(event)

    def key_press_tile(self, event):
        if event.key in ["right", "left", "up", "down"]:
            if self.current_artist is not None:
                self.current_artist.move_image(event.key)
        elif event.key in ["shift+right", "shift+left", "shift+up", "shift+down"]:
            if self.current_artist is not None:
                self.current_artist.move_image(event.key.split('+')[1], 0.25)
        elif event.key.lower() == "n":
            i = self.current_artist_index
            self.current_artist_index = (i + 1) % len(self.tile_artists)
        elif event.key.lower() == "p":
            i = len(self.tile_artists) + 1
            self.current_artist_index = (i - 1) % len(self.tile_artists)

    def key_press_point_plot(self, event):
        if event.key.startswith('shift+'):
            direction = event.key.split('+')[1]
            scale = 0.025
        else:
            direction = event.key
            scale = 0.1

        if direction in ["right", "left"]:
            lb, ub = self.axes.get_xlim()
            shift = (ub-lb) * scale * (1 if direction == 'right' else -1)
            self.axes.set_xlim(lb + shift, ub + shift)
        elif direction in ["up", "down"]:
            lb, ub = self.axes.get_ylim()
            shift = (ub-lb) * scale * (1 if event.key == 'up' else -1)
            self.axes.set_ylim(lb + shift, ub + shift)
        self.redraw()

    def button_press(self, event):
        if event.button == MouseButton.LEFT and event.xdata is not None:
            self.start_pan(event)
        elif self.tool != 'tile':
            self.button_press_point_plot(event)

    def button_release_tile(self, event):
        if event.button == MouseButton.LEFT and event.xdata is not None:
            for i, artist in enumerate(self.tile_artists.values()):
                if artist.contains(event.xdata, event.ydata):
                    self.current_artist_index = i
                    break

    def button_press_point_plot(self, event):
        if event.button != MouseButton.RIGHT:
            return
        if self.cells == 'Extra' and self.tool != 'cells':
            # Special case. I don't want to add spiral/exclude regions to extra
            # cells data structure for now.
            return
        if event.key == 'control' and event.xdata is not None:
            if self.tool == 'spiral':
                self.point_artists[self.cells, 'spiral'].set_origin(event.xdata, event.ydata)
        elif event.key == "shift" and event.xdata is not None:
            if self.tool == 'cells':
                self.point_artists[self.cells, 'cells'].remove_point(event.xdata, event.ydata)
            elif self.tool == 'spiral':
                self.point_artists[self.cells, 'spiral'].remove_point(event.xdata, event.ydata)
            elif self.tool == 'exclude':
                self.point_artists[self.cells, 'spiral'].remove_exclude(event.xdata, event.ydata)
        elif event.xdata is not None:
            if self.tool == 'cells':
                self.point_artists[self.cells, 'cells'].add_point(event.xdata, event.ydata)
            elif self.tool == 'spiral':
                self.point_artists[self.cells, 'spiral'].add_point(event.xdata, event.ydata)
            elif self.tool == 'exclude':
                if self.drag_event is None:
                    self.start_drag_exclude(event)
                else:
                    self.end_drag_exclude(event, keep=True)

    @observe('tool', 'cells')
    def _reset_drag(self, event):
        self.drag_event = None
        if self.current_spiral_artist is not None:
            self.current_spiral_artist.start_drag = None
            self.current_spiral_artist.end_drag = None

    def button_release(self, event):
        if event.button == MouseButton.LEFT:
            if not self.pan_performed:
                self.button_release_tile(event)
            self.end_pan(event)
        elif event.button == MouseButton.RIGHT:
            if self.tool == 'tile':
                self.drag_event = None

    def motion(self, event):
        if self.pan_event is not None:
            self.motion_pan(event)
        elif self.drag_event is not None:
            self.motion_drag(event)

    def start_pan(self, event):
        self.pan_event = event
        self.pan_performed = False
        self.pan_xlim = self.axes.get_xlim()
        self.pan_ylim = self.axes.get_ylim()

    def motion_pan(self, event):
        if event.xdata is None:
            return
        if self.pan_event is None:
            return
        dx = event.xdata - self.pan_event.xdata
        dy = event.ydata - self.pan_event.ydata
        self.pan_xlim -= dx
        self.pan_ylim -= dy
        self.axes.set_xlim(self.pan_xlim)
        self.axes.set_ylim(self.pan_ylim)
        self.pan_performed = True
        self.redraw()

    def end_pan(self, event):
        self.pan_event = None

    def start_drag_exclude(self, event):
        self.drag_event = event
        self.current_spiral_artist.start_exclude(event.xdata, event.ydata)

    def start_drag_tile(self, event):
        self.drag_event = event

    def motion_drag(self, event):
        if event.xdata is None:
            if self.tool == 'exclude':
                self.end_drag_exclude(event, keep=False)
        elif self.tool == 'tile' and self.current_artist is not None:
            dx = event.xdata - self.drag_event.xdata
            dy = event.ydata - self.drag_event.ydata
            self.current_artist.drag_image(dx, dy)
            self.drag_event = event
        else:
            self.current_spiral_artist.update_exclude(event.xdata, event.ydata)

    def end_drag_exclude(self, event, keep):
        self.current_spiral_artist.end_exclude(keep=keep)
        self.drag_event = None

    def end_drag_tile(self, event):
        self.drag_event = None

    def scroll(self, event):
        """
        This zooms in without shifting the center point
        """
        if event.xdata is None:
            return

        base_scale = 1.1

        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()
        cur_xrange = cur_xlim[1] - cur_xlim[0]
        cur_yrange = cur_ylim[1] - cur_ylim[0]

        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        xfrac = (xdata - cur_xlim[0]) / cur_xrange
        yfrac = (ydata - cur_ylim[0]) / cur_yrange

        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1

        # set new limits
        new_xrange = cur_xrange * scale_factor
        new_xlim = [xdata - xfrac * new_xrange, xdata + (1 - xfrac) * new_xrange]

        new_yrange = cur_yrange * scale_factor
        new_ylim = [ydata - yfrac * new_yrange, ydata + (1 - yfrac) * new_yrange]
        self.axes.set_xlim(new_xlim)
        self.axes.set_ylim(new_ylim)
        self.redraw()

    def set_display_mode(self, display_mode, all_tiles=False):
        if all_tiles:
            for artist in self.tile_artists.values():
                artist.display_mode = display_mode
        elif self.current_artist is not None:
            self.current_artist.display_mode = display_mode

    def set_channel_visible(self, channel_name, visible, all_tiles=False):
        if all_tiles:
            for artist in self.tile_artists.values():
                artist.set_channel_visible(channel_name, visible)
        elif self.current_artist is not None:
            self.current_artist.set_channel_visible(channel_name, visible)

    def set_channel_min_value(self, channel_name, low_value, all_tiles=False):
        if all_tiles:
            for artist in self.tile_artists.values():
                artist.set_channel_min_value(channel_name, low_value)
        elif self.current_artist is not None:
            self.current_artist.set_channel_min_value(channel_name, low_value)

    def set_channel_max_value(self, channel_name, high_value, all_tiles=False):
        if all_tiles:
            for artist in self.tile_artists.values():
                artist.set_channel_max_value(channel_name, high_value)
        elif self.current_artist is not None:
            self.current_artist.set_channel_max_value(channel_name, high_value)

    def set_z_slice(self, z_slice, all_tiles=False):
        if all_tiles:
            for artist in self.tile_artists.values():
                artist.z_slice = z_slice
        elif self.current_artist is not None:
            self.current_artist.z_slice = z_slice

    def get_state(self):
        artist_states = {k: a.get_state() for k, a in self.tile_artists.items()}
        point_artist_states = {':'.join(k): a.get_state() for k, a in self.point_artists.items()}
        return {
            "cells": self.cells,
            "tool": self.tool,
            "artists": artist_states,
            "point_artists": point_artist_states,
        }

    def set_state(self, state):
        for k, s in state["artists"].items():
            self.tile_artists[k].set_state(s)
        for k, s in state["point_artists"].items():
            self.point_artists[tuple(k.split(':'))].set_state(s)
        self.set_interaction_mode(state["cells"], state["tool"])

    def get_full_state(self):
        return deepcopy({
            "data": self.piece.get_state(),
            "view": self.get_state(),
        })

    def save_state(self):
        state = self.get_full_state()
        self.reader.save_state(self.piece, state)
        self.saved_state = state
        self.update()

    def load_state(self):
        state = self.reader.load_state(self.piece)
        self.piece.set_state(state['data'])
        self.saved_state = state
        self.update()

    def redraw(self):
        self.figure.canvas.draw()

    def redraw_if_needed(self):
        if self.needs_redraw:
            self.redraw()
            self.needs_redraw = False
