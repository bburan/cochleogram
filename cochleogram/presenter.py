import json
from atom.api import (
    Atom,
    Bool,
    Dict,
    Enum,
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

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib import (
    patheffects,
    ticker,
)

import numpy as np
from scipy import interpolate

from cochleogram.model import Piece, Points, Tile
from cochleogram.util import shortest_path


class PointPlot(Atom):

    artist = Value()
    spline_artist = Value()
    axes = Value()
    points = Typed(Points)
    name = Str()
    visible = Bool(True)
    has_nodes = Bool(False)

    def __init__(self, axes, points, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes
        (self.artist,) = axes.plot([], [], "ko", mec="w", mew=1, zorder=100)
        self.points = points

    def get_state(self):
        return {}

    def set_state(self, state):
        self.redraw()

    def add_point(self, x, y):
        if self.points.add_node(x, y):
            self.redraw()

    def remove_point(self, x, y):
        if self.points.remove_node(x, y):
            self.redraw()

    def redraw(self, event=None):
        nodes = self.points.get_nodes()
        self.has_nodes = len(nodes[0]) > 0
        self.artist.set_data(*nodes)

    def _observe_visible(self, event):
        self.artist.set_visible(self.visible)


class LinePlot(PointPlot):

    spline_artist = Value()
    has_spline = Bool(False)

    def __init__(self, axes, points, **kwargs):
        super().__init__(axes, points, **kwargs)
        spline_effect = [
            patheffects.Stroke(linewidth=3, foreground="white"),
            patheffects.Normal(),
        ]
        (self.spline_artist,) = axes.plot(
            [], [], "k-", zorder=90, path_effects=spline_effect
        )

    def redraw(self):
        super().redraw()
        xi, yi = self.points.interpolate()
        self.has_spline = len(xi) > 0
        self.spline_artist.set_data(xi, yi)

    def _observe_visible(self, event):
        super()._observe_visible(event)
        self.spline_artist.set_visible(self.visible)


class ImagePlot(Atom):

    alpha = Float(0.75)
    zorder = Int(10)

    display_mode = Enum("projection", "slice")
    display_channel = Enum("All", "CtBP2", "MyosinVIIa")
    extent = Tuple()
    z_slice = Int(0)
    z_slice_min = Int(0)
    z_slice_max = Int(0)
    shift = Float()

    tile = Typed(Tile)
    artist = Value()
    axes = Value()

    def get_state(self):
        return {
            "alpha": self.alpha,
            "zorder": self.zorder,
            "display_mode": self.display_mode,
            "display_channel": self.display_channel,
            "z_slice": self.z_slice,
            "z_slice_min": self.z_slice_min,
            "z_slice_max": self.z_slice_max,
            "shift": self.shift,
        }

    def set_state(self, state):
        self.alpha = state["alpha"]
        self.zorder = state["zorder"]
        self.display_mode = state["display_mode"]
        self.display_channel = state["display_channel"]
        self.z_slice = state["z_slice"]
        self.z_slice_min = state["z_slice_min"]
        self.z_slice_max = state["z_slice_max"]
        self.shift = state["shift"]
        self.move_image()
        self.update_image()

    def __init__(self, axes, tile, **kwargs):
        super().__init__(**kwargs)
        self.tile = tile
        self.axes = axes
        fmt = ticker.FuncFormatter(lambda x, i: f'{x*1e3:.2f}')
        self.axes.xaxis.set_major_formatter(fmt)
        self.axes.yaxis.set_major_formatter(fmt)
        self.artist = axes.imshow(np.zeros((0, 0)), origin="lower")
        self.z_slice_max = self.tile.image.shape[2] - 1
        self.z_slice = self.tile.image.shape[2] // 2
        self.shift = self.tile.info["scaling"][0] * 5
        self.move_image()
        self.update_image()

    def _observe_alpha(self, event):
        self.artist.set_alpha(self.alpha)

    def _observe_zorder(self, event):
        self.artist.set_zorder(self.zorder)

    def move_image(self, direction=None):
        extent = np.array(self.tile.extent)
        if direction == "up":
            extent[2:] += self.shift
        elif direction == "down":
            extent[2:] -= self.shift
        elif direction == "left":
            extent[:2] -= self.shift
        elif direction == "right":
            extent[:2] += self.shift
        elif direction is None:
            pass
        self.tile.extent = tuple(extent.tolist())
        self.artist.set_extent(self.tile.extent)

    @observe("z_slice", "display_mode", "display_channel")
    def update_image(self, event=None):
        if self.display_channel == "All":
            image = self.tile.image
        elif self.display_channel == "CtBP2":
            image = self.tile.image[:, :, :, 0]
        elif self.display_channel == "MyosinVIIa":
            image = self.tile.image[:, :, :, 1]
        if self.display_mode == "projection":
            image = np.max(image, axis=2)
        elif self.display_mode == "slice":
            image = image[:, :, self.z_slice]
        self.artist.set_data(image[::-1, ::-1])

    def contains(self, x, y):
        return self.tile.contains(x, y)


class Presenter(Atom):

    # Tile artists
    artists = Dict()
    current_artist_index = Int()
    current_artist = Property()

    # For spirals and cells
    point_artists = Dict()
    current_point_artist = Property()
    current_spiral_artist = Value()
    current_cells_artist = Value()

    figure = Typed(Figure)
    axes = Typed(Axes)
    piece = Typed(Piece)

    highlight_selected = Bool(True)
    alpha_selected = Float(0.75)
    alpha_unselected = Float(0.25)
    zorder_selected = Int(20)
    zorder_unselected = Int(10)

    interaction_mode = Enum("tiles", "IHC", "OHC1", "OHC2", "OHC3")
    interaction_submode = Enum("spiral", "cells")

    pan_event = Value()
    pan_xlim = Value()
    pan_ylim = Value()

    spiral_empty = Bool(True)
    spiral_ready = Bool(False)
    cells_empty = Bool(True)

    def __init__(self, piece, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piece = piece
        self.artists = {
            t.source.name: ImagePlot(self.axes, t) for t in self.piece.tiles
        }
        self.current_artist_index = 0
        for key in ('IHC', 'OHC1', 'OHC2', 'OHC3'):
            self.point_artists[key, 'cells'] = PointPlot(self.axes, self.piece.cells[key], name=key)
            self.point_artists[key, 'spiral'] = LinePlot(self.axes, self.piece.spirals[key], name=key)

        # Not sure why this is necessary
        self.axes.axis(self.piece.get_image_extent())

    def _default_figure(self):
        return Figure()

    def _default_axes(self):
        return self.figure.add_axes([0.1, 0.1, 0.8, 0.8])

    def _get_current_artist(self):
        return list(self.artists.values())[self.current_artist_index]

    def _get_current_point_artist(self):
        return self.point_artists[self.interaction_mode, self.interaction_submode]

    @observe("highlight_selected",)
    def update_highlight(self, event=None):
        alpha = self.alpha_unselected if self.highlight_selected else 1
        for artist in self.artists.values():
            artist.zorder = self.zorder_unselected
            artist.alpha = alpha
        if self.highlight_selected:
            self.current_artist.alpha = self.alpha_selected
        self.current_artist.zorder = self.zorder_selected
        self.redraw()

    def button_press(self, event):
        if event.button == MouseButton.RIGHT:
            self.start_pan(event)
        elif self.interaction_mode == 'tiles':
            self.button_press_tiles(event)
        else:
            self.button_press_point_plot(event)

    def button_press_tiles(self, event):
        if event.button == MouseButton.LEFT:
            for i, artist in enumerate(self.artists.values()):
                if artist.contains(event.xdata, event.ydata):
                    self.current_artist_index = i
                    break

    def button_press_point_plot(self, event):
        if event.button != MouseButton.LEFT:
            return
        if event.key == "shift":
            self.current_point_artist.remove_point(event.xdata, event.ydata)
        elif event.key is None:
            self.current_point_artist.add_point(event.xdata, event.ydata)
        self.redraw()

    def button_release(self, event):
        if event.button == MouseButton.RIGHT:
            self.end_pan(event)

    @observe('interaction_mode', 'interaction_submode')
    def _update_plots(self, event=None):
        for artist in self.point_artists.values():
            artist.visible = False
        if self.interaction_mode == 'tiles':
            self.current_spiral_artist = None
            self.current_cells_artist = None
        else:
            self.current_spiral_artist = self.point_artists[self.interaction_mode, 'spiral']
            self.current_cells_artist = self.point_artists[self.interaction_mode, 'cells']
            self.current_point_artist.visible = True
        self.redraw()

    def set_interaction_mode(self, mode, submode):
        self.interaction_mode = mode
        self.interaction_submode = submode

    def action_guess_cells(self):
        print(self.interaction_mode)
        self.piece.guess_cells(self.interaction_mode)
        print('setting interaction mode')
        self.current_point_artist.redraw()
        self.redraw()
        self.set_interaction_mode(self.interaction_mode, 'cells')

    def action_clear_cells(self):
        self.piece.clear_cells(self.interaction_mode)
        self.set_interaction_mode(self.interaction_mode, 'cells')
        self.current_point_artist.redraw()
        self.redraw()

    def action_clear_spiral(self):
        self.piece.clear_spiral(self.interaction_mode)
        self.set_interaction_mode(self.interaction_mode, 'spiral')
        self.current_point_artist.redraw()
        self.redraw()

    def action_clone_spiral(self, to_spiral, distance):
        print(distance)
        xn, yn = self.piece.spirals[self.interaction_mode].expand_nodes(distance)
        self.piece.spirals[to_spiral].set_nodes(xn, yn)
        self.point_artists[to_spiral, 'spiral'].redraw()
        #self.current_point_artist.redraw()
        #self.redraw()

    def key_press(self, event):
        if self.interaction_mode == 'tiles':
            return self.key_press_tiles(event)

    def key_press_tiles(self, event):
        if event.key in ["right", "left", "up", "down"]:
            self.current_artist.move_image(event.key)
            self.redraw()
        elif event.key.lower() == "n":
            self.current_artist_index = int(
                np.clip(self.current_artist_index + 1, 0, len(self.artists) - 1)
            )
        elif event.key.lower() == "p":
            self.current_artist_index = int(
                np.clip(self.current_artist_index - 1, 0, len(self.artists) - 1)
            )

    def _observe_current_artist_index(self, event):
        self.update_highlight()

    def motion(self, event):
        self.pan(event)

    def start_pan(self, event):
        self.pan_event = event
        self.pan_xlim = self.axes.get_xlim()
        self.pan_ylim = self.axes.get_ylim()

    def end_pan(self, event):
        self.pan_event = None

    def pan(self, event):
        if self.pan_event is None:
            return
        dx = event.xdata - self.pan_event.xdata
        dy = event.ydata - self.pan_event.ydata
        self.pan_xlim -= dx
        self.pan_ylim -= dy
        self.axes.set_xlim(self.pan_xlim)
        self.axes.set_ylim(self.pan_ylim)
        self.redraw()

    def scroll(self, event):
        """
        This zooms in without shifting the center point
        """
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
            for artist in self.artists.values():
                artist.display_mode = display_mode
        else:
            self.current_artist.display_mode = display_mode
        self.redraw()

    def set_display_channel(self, display_channel, all_tiles=False):
        if all_tiles:
            for artist in self.artists.values():
                artist.display_channel = display_channel
        else:
            self.current_artist.display_channel = display_channel
        self.redraw()

    def set_z_slice(self, z_slice, all_tiles=False):
        if all_tiles:
            for artist in self.artists.values():
                artist.z_slice = z_slice
        else:
            self.current_artist.z_slice = z_slice
        self.redraw()

    def get_state(self):
        artist_states = {k: a.get_state() for k, a in self.artists.items()}
        point_artist_states = {':'.join(k): a.get_state() for k, a in self.point_artists.items()}
        return {
            "interaction_mode": self.interaction_mode,
            "interaction_submode": self.interaction_submode,
            "artists": artist_states,
            "point_artists": point_artist_states,
        }

    def set_state(self, state):
        for k, s in state["artists"].items():
            self.artists[k].set_state(s)
        for k, s in state["point_artists"].items():
            self.point_artists[tuple(k.split(':'))].set_state(s)
        self.set_interaction_mode(state["interaction_mode"],
                                  state["interaction_submode"])

    def save_state(self):
        state = {
            "data": self.piece.get_state(),
            "view": self.get_state(),
        }
        state_filename = self.piece.path / f"piece_{self.piece.piece}.json"
        state_filename.write_text(json.dumps(state, indent=4))

    def load_state(self):
        state_filename = self.piece.path / f"piece_{self.piece.piece}.json"
        state = json.loads(state_filename.read_text())
        self.piece.set_state(state['data'])
        self.set_state(state['view'])
        self._update_plots()

    def redraw(self):
        self.figure.canvas.draw()
