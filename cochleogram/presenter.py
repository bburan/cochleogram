from copy import deepcopy
import time

import json
from atom.api import (
    Atom,
    Bool,
    Dict,
    Enum,
    Event,
    Float,
    Instance,
    Int,
    List,
    observe,
    Property,
    set_default,
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
from matplotlib import ticker
from matplotlib import patheffects as pe

from matplotlib import patches as mpatches
from matplotlib import path as mpath
from matplotlib import transforms as T

import numpy as np
from scipy import interpolate

from cochleogram.config import CELLS, CELL_COLORS, CELL_KEY_MAP, TOOL_KEY_MAP
from cochleogram.model import ChannelConfig, Piece, Points, Tile
from cochleogram.readers import BaseReader
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
    active = Bool(True)
    has_nodes = Bool(False)

    updated = Event()
    needs_redraw = Bool(False)

    base_color = Value('black')
    artist_styles = Dict()

    def _default_artist_styles(self):
        return {
            'active': [
                pe.PathPatchEffect(facecolor=self.base_color, edgecolor='white', linewidth=1),
            ],
            'inactive': [
                pe.PathPatchEffect(facecolor=self.base_color, edgecolor='none', alpha=0.75),
            ],
        }

    def __init__(self, axes, points, **kwargs):
        super().__init__(axes=axes, points=points, **kwargs)
        self.artist, = self.axes.plot([], [], "o", zorder=100, color='none')
        self.points.observe('updated', self.request_redraw)

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

    @observe("active")
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
        style = 'active' if self.active else 'inactive'
        self.artist.set_path_effects(self.artist_styles[style])
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

    #: Is the exclusion region active?
    exclude_active = Bool(False)

    start_drag = Value()
    end_drag = Value()

    spline_artist_styles = Dict()
    origin_artist_styles = Dict()

    def _default_artist_styles(self):
        return {
            'active': [
                pe.PathPatchEffect(facecolor=self.base_color, edgecolor='white', linewidth=1),
            ],
            'inactive': [],
        }

    def _default_spline_artist_styles(self):
        return {
            'active': [
                pe.Stroke(linewidth=3, foreground='white'),
                pe.Stroke(linewidth=1, foreground=self.base_color),
            ],
            'inactive': [
                pe.Stroke(foreground=self.base_color, linewidth=1, alpha=0.75),
            ],
        }

    def _default_origin_artist_styles(self):
        return {
            'active': [
                pe.PathPatchEffect(facecolor='FireBrick', edgecolor='white', linewidth=1),
            ],
            'inactive': [],
        }

    def __init__(self, axes, points, **kwargs):
        super().__init__(axes, points, **kwargs)
        self.spline_artist, = axes.plot([], [], "-", zorder=90, color='none')
        self.origin_artist, = axes.plot([], [], "o", color='none', zorder=90, ms=10)

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

    def _observe_exclude_active(self, event=False):
        self.needs_redraw = True
        deferred_call(self.redraw_if_needed)

    def redraw(self, event=None):
        super().redraw()
        style = 'active' if self.active else 'inactive'
        self.spline_artist.set_path_effects(self.spline_artist_styles[style])
        self.origin_artist.set_path_effects(self.origin_artist_styles[style])

        if self.has_nodes:
            nodes = self.points.get_nodes()
            self.origin_artist.set_data(nodes[0][0], nodes[1][0])
        else:
            self.origin_artist.set_data([], [])

        xi, yi = self.points.interpolate()
        self.has_spline = len(xi) > 0
        self.spline_artist.set_data(xi, yi)
        self.new_exclude_artist.set_visible(self.active)

        self.has_exclusion = len(self.points.exclude) > 0
        path = make_plot_path(self.points, self.points.exclude)

        self.exclude_artist.set_path(path)
        self.exclude_artist.set_visible(self.exclude_active)

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
    visible_channels = Property()
    extent = Tuple()
    z_slice = Int(0)
    z_slice_min = Int(0)
    z_slice_max = Int(0)
    shift = Float()

    tile = Typed(Tile)
    artist = Value()
    rectangle = Value()
    axes = Value()
    auto_rotate = Bool(True)
    rotation_transform = Value()
    transform = Value()

    updated = Event()
    needs_redraw = Bool(False)

    def _get_visible_channels(self):
        return [c.name for c in self.channel_config.values() if c.visible]

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
        if self.auto_rotate:
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


class BasePresenter(Atom):

    #: Label of cell being marked
    cells = Str()

    #: List of available cells
    available_cells = Tuple()

    def _default_available_cells(self):
        return CELLS

    def _default_cells(self):
        return self.available_cells[0]

    #: Active tool
    tool = Str()

    #: List of available tools
    available_tools = Tuple()

    def _default_tool(self):
        if 'spiral' in self.available_tools:
            return 'spiral'
        else:
            return self.available_tools[0]

    #: List of valid key shortcuts
    valid_keys = List()

    def _default_valid_keys(self):
        tkm_inv = {v: k for k, v in TOOL_KEY_MAP}
        ckm_inv = {v: k for k, v in CELL_KEY_MAP}
        tk = [tkm_inv[t] for t in self.available_tools]
        ck = [ckm_inv[t] for t in self.available_cells]
        return tuple(tk + ck)

    #: Interface to help read data
    reader = Instance(BaseReader)

    #: Parent figure of the axes
    figure = Typed(Figure)

    #: Axes on which all artists will be rendered
    axes = Typed(Axes)

    #: Object being presented (instance of a class in cochleogram.model)
    obj = Value()

    #: State as saved to file (used for checking against present state to
    #: determine if there are unsaved changes)
    saved_state = Dict()

    #: Flag indicating whether there are unsaved changes
    unsaved_changes = Bool(False)

    #: Flag indicating whether the figure needs to be redrawn
    needs_redraw = Bool(False)

    #: Currently selected tile artist
    current_artist = Value()

    #: Dictionary of all tile artists
    tile_artists = Dict()

    #: Minimum z-slice across all tiles in image
    z_min = Property()

    #: Maximum z-slice across all tiles in image
    z_max = Property()

    #: Starting pan event
    pan_event = Value(None)
    pan_xlim = Value()
    pan_ylim = Value()

    #: True if we actually had a pan event. This allows us to distinguish
    #: between clicks that select a tile vs. clicks that are intended to start
    #: a pan.
    pan_performed = Bool(False)

    drag_event = Value(None)
    drag_x = Value()
    drag_y = Value()

    # For spirals and cells
    point_artists = Dict()
    current_spiral_artist = Value()
    current_cells_artist = Value()

    current_artist_index = Int()

    #: Track timestamp of last scroll event recieved to ensure that we don't
    #: zoom too quickly.
    last_scroll_time = Value(time.time())

    #: If True, rotate tiles so they represent the orientation they were imaged
    #: on the confocal.
    rotate_tiles = Bool(True)

    def __init__(self, obj, reader, **kwargs):
        super().__init__(obj=obj, reader=reader, **kwargs)
        self.tile_artists = {t.source: ImagePlot(self.axes, t, auto_rotate=self.rotate_tiles) \
                             for t in self.obj}

        for artist in self.tile_artists.values():
            artist.observe('updated', self.update)

        for key in self.available_cells:
            color = CELL_COLORS[key]
            cells = PointPlot(self.axes, self.obj.cells[key], name=key, base_color=color)
            spiral = LinePlot(self.axes, self.obj.spirals[key], name=key, base_color=color)
            cells.observe('updated', self.update)
            spiral.observe('updated', self.update)
            self.point_artists[key, 'cells'] = cells
            self.point_artists[key, 'spiral'] = spiral

        # Needs to be set to force a change notification that sets the current
        # artist.
        self.current_artist_index = 0

        # This is necessary because `imshow` will override some axis settings.
        # We need to set them back to what we want.
        self.axes.axis('equal')
        self.axes.axis(self.obj.get_image_extent())
        self.load_state()

    def _default_saved_state(self):
        return self.get_full_state()

    ###################################################################################
    # Code for handling events from Matplotlib
    def motion(self, event):
        if self.pan_event is not None:
            self.motion_pan(event)
        elif self.drag_event is not None:
            self.motion_drag(event)

    def motion_drag(self, event):
        raise NotImplementedError

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
        self.update()

    def end_pan(self, event):
        self.pan_event = None

    def button_release(self, event):
        if event.button == MouseButton.LEFT:
            self.left_button_release(event)
        elif event.button == MouseButton.RIGHT:
            self.right_button_release(event)

    def left_button_release(self, event):
        self.end_pan(event)

    def right_button_release(self, event):
        pass

    def button_press(self, event):
        if event.button == MouseButton.LEFT:
            self.left_button_press(event)
        elif event.button == MouseButton.RIGHT:
            self.right_button_press(event)

    def left_button_press(self, event):
        if event.xdata is not None:
            self.start_pan(event)

    def right_button_press(self, event):
        pass

    def scroll_zaxis(self, step):
        if self.current_artist.display_mode == 'projection':
            if step == 'down':
                z = self.current_artist.z_slice_max
            else:
                z = self.current_artist.z_slice_min
        else:
            if step == 'down':
                z = self.current_artist.z_slice - 1
            else:
                z = self.current_artist.z_slice + 1
        lb, ub = self.current_artist.z_slice_min, self.current_artist.z_slice_max
        if lb <= z <= ub:
            self.current_artist.display_mode = 'slice'
            self.current_artist.z_slice = z
        else:
            self.current_artist.display_mode = 'projection'
        self.update()

    def scroll(self, event):
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
        self.update()

    def _get_z_min(self):
        return min(a.z_slice_min for a in self.tile_artists.values())

    def _get_z_max(self):
        return min(a.z_slice_max for a in self.tile_artists.values())

    def _default_figure(self):
        return Figure()

    def _default_axes(self):
        return self.figure.add_axes([0, 0, 1, 1])

    def save_state(self):
        state = self.get_full_state()
        self.reader.save_state(self.obj, state)
        self.saved_state = state
        self.update()

    def load_state(self):
        try:
            state = self.reader.load_state(self.obj)
            self.obj.set_state(state['data'])
            self.saved_state = state
            self.update()
        except IOError:
            pass

    def update(self, event=None):
        self.check_for_changes()
        self.needs_redraw = True
        deferred_call(self.redraw_if_needed)

    def redraw(self):
        self.figure.canvas.draw()

    def redraw_if_needed(self):
        if self.needs_redraw:
            self.redraw()
            self.needs_redraw = False

    def check_for_changes(self):
        raise NotImplementedError

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

    @observe('cells', 'tool')
    def _update_plots(self, event=None):
        for artist in self.point_artists.values():
            artist.active = False
            if hasattr(artist, 'exclude_active'):
                artist.exclude_active = False
        if self.tool == 'tile':
            self.current_spiral_artist = None
            self.current_cells_artist = None
        else:
            self.current_spiral_artist = self.point_artists[self.cells, 'spiral']
            self.current_cells_artist = self.point_artists[self.cells, 'cells']
            self.current_spiral_artist.exclude_active = True
            if self.tool in ('spiral', 'exclude'):
                self.current_spiral_artist.active = True
            else:
                self.current_cells_artist.active = True

    def action_guess_cells(self, width, spacing, channel):
        z_slice = self.current_artist.z_slice if self.current_artist.display_mode == 'slice' else None
        n = self.obj.guess_cells(self.cells, width, spacing, channel, z_slice)
        self.set_interaction_mode(None, 'cells')
        return n

    def action_clear_cells(self):
        self.obj.clear_cells(self.cells)
        self.set_interaction_mode(None, 'cells')

    def action_clear_spiral(self):
        self.obj.clear_spiral(self.cells)
        self.set_interaction_mode(None, 'spiral')

    def set_interaction_mode(self, cells=None, tool=None):
        if cells is not None:
            self.cells = cells
        if tool is not None:
            self.tool = tool

    def button_press_point_plot(self, event):
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

    def key_press(self, event):
        key = event.key.lower()
        if (t := TOOL_KEY_MAP.get(key, None)) is not None:
            deferred_call(self.set_interaction_mode, None, t)
            return True
        if (c := CELL_KEY_MAP.get(key, None)) is not None:
            deferred_call(self.set_interaction_mode, c, None)
            return True
        return False

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
            "data": self.obj.get_state(),
            "view": self.get_state(),
        })

    def _observe_current_artist_index(self, event):
        self.current_artist = list(self.tile_artists.values())[self.current_artist_index]

    def _observe_saved_state(self, event):
        self.check_for_changes()

    def check_for_changes(self):
        saved = self.saved_state['data'].copy()
        unsaved = self.get_full_state()['data']
        self.unsaved_changes = saved != unsaved


class CellCountPresenter(BasePresenter):

    available_tools = set_default(('spiral', 'cells'))
    rotate_tiles = set_default(False)

    def right_button_press(self, event):
        self.button_press_point_plot(event)

    def set_interaction_mode(self, cells=None, tool=None):
        if cells is not None and tool is None:
            if cells != 'Extra' and not self.point_artists[cells, 'spiral'].has_nodes:
                tool = 'spiral'
            else:
                tool = 'cells'
        super().set_interaction_mode(cells, tool)

    def key_press(self, event):
        # If this returns True, event was handled
        if super().key_press(event):
            return
        if self.current_artist is None:
            return
        if event.key in ["up", "down"]:
            self.scroll_zaxis(event.key)
        self.update()


class CochleogramPresenter(BasePresenter):

    highlight_selected = Bool(False)
    alpha_selected = Float(0.50)
    alpha_unselected = Float(0.50)
    zorder_selected = Int(20)
    zorder_unselected = Int(10)
    rotate_tiles = set_default(True)

    available_tools = set_default(("tile", "spiral", "exclude", "cells"))

    def _observe_saved_state(self, event):
        self.check_for_changes()

    def check_for_changes(self):
        saved = self.saved_state['data'].copy()
        saved.pop('copied_from', None)
        unsaved = self.get_full_state()['data']
        unsaved.pop('copied_from', None)
        self.unsaved_changes = saved != unsaved

    def _observe_current_artist_index(self, event):
        super()._observe_current_artist_index(event)
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
        self.update()

    def action_auto_align_tiles(self):
        self.obj.align_tiles(self.current_artist.visible_channels)

    def action_clone_spiral(self, to_spiral, distance):
        xn, yn = self.obj.spirals[self.cells].expand_nodes(distance)
        self.obj.spirals[to_spiral].set_nodes(xn, yn)

    def action_copy_exclusion(self, to_spiral):
        if not self.point_artists[to_spiral, 'spiral'].has_spline:
            raise ValueError(f'Must create spiral for {to_spiral} first')
        for s, e in self.obj.spirals[self.cells].exclude:
            self.obj.spirals[to_spiral].add_exclude(s, e)

    def action_merge_exclusion(self, *spirals):
        exclude = []
        for spiral in spirals:
            if not self.point_artists[spiral, 'spiral'].has_spline:
                raise ValueError(f'Must create spiral for {spiral} first')
            exclude.extend(self.obj.spirals[spiral].exclude)
        for spiral in spirals:
            self.obj.spirals[spiral].exclude = exclude
            self.obj.spirals[spiral].simplify_exclude()

    def action_simplify_exclusion(self, *spirals):
        for spiral in spirals:
            self.obj.spirals[spiral].simplify_exclude()

    def key_press(self, event):
        # If this returns True, event was handled
        if super().key_press(event):
            return
        elif (event.key.lower() == 'escape') and (self.drag_event is not None) and (self.tool == 'exclude'):
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
        self.update()

    def right_button_press(self, event):
        if self.tool != 'tile':
            self.button_press_point_plot(event)

    def left_button_release(self, event):
        if not self.pan_performed:
            self.button_release_tile(event)
        super().left_button_release(event)

    def button_release_tile(self, event):
        if event.button == MouseButton.LEFT and event.xdata is not None:
            for i, artist in enumerate(self.tile_artists.values()):
                if artist.contains(event.xdata, event.ydata):
                    self.current_artist_index = i
                    break

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
