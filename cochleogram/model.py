import logging
log = logging.getLogger(__name__)

import json
from pathlib import Path
import pickle
import re

from atom.api import Atom, Dict, Event, Float, Int, List, Typed
from matplotlib import colors
import numpy as np
import pandas as pd

from psiaudio.util import octave_space
from scipy import interpolate
from scipy import ndimage
from scipy import signal
from raster_geometry import sphere

from cochleogram import util


class Points(Atom):

    x = List()
    y = List()
    origin = Int()
    exclude = List()

    updated = Event()

    def __init__(self, x=None, y=None, origin=0, exclude=None):
        self.x = [] if x is None else x
        self.y = [] if y is None else y
        self.origin = origin
        self.exclude = [] if exclude is None else exclude

    def expand_nodes(self, distance):
        '''
        Expand the spiral outward by the given distance
        '''
        # The algorithm generates an interpolated spline that can be used to
        # calculate the angle at any given point along the curve. We can then
        # add pi/2 (i.e., 90 degrees) to get the angel of the line that's
        # perpendicular to the spline at that particular point.
        x, y = self.interpolate(resolution=0.01)
        xn, yn = self.get_nodes()
        v = x + y * 1j
        vn = np.array(xn) + np.array(yn) * 1j
        a = np.angle(np.diff(v)) + np.pi / 2

        # Once we have the angles of lines perpendicular to the spiral at all
        # the interpolated points, we need to find the interpolated points
        # closest to our actual nodes.
        i = np.abs(v[1:] - vn[:, np.newaxis]).argmin(axis=1)
        a = a[i]
        dx = distance * np.cos(a)
        dy = distance * np.sin(a)

        return xn + dx, yn + dy

    def get_nodes(self):
        """
        Simple algorithm that assumes that the next "nearest" node is the one
        we want to draw a path through. This avoids trying to solve the
        complete traveling salesman problem which is NP-hard.
        """
        i = self.origin
        nodes = list(zip(self.x, self.y))
        path = []
        while len(nodes) > 1:
            n = nodes.pop(i)
            path.append(n)
            d = np.sqrt(np.sum((np.array(nodes) - n) ** 2, axis=1))
            i = np.argmin(d)
        path.extend(nodes)
        if path:
            return list(zip(*path))
        return [(), ()]

    def direction(self):
        x, y = self.interpolate()
        return util.arc_direction(x, y)

    def interpolate(self, degree=3, smoothing=0, resolution=0.001):
        nodes = self.get_nodes()
        if len(nodes[0]) <= 3:
            return [], []
        tck, u = interpolate.splprep(nodes, k=degree, s=smoothing)
        x = np.arange(0, 1 + resolution, resolution)
        xi, yi = interpolate.splev(x, tck, der=0)
        return xi, yi

    def length(self, degree=3, smoothing=0, resolution=0.001):
        nodes = self.get_nodes()
        if len(nodes[0]) <= 3:
            return np.nan
        tck, u = interpolate.splprep(nodes, k=degree, s=smoothing)
        x = np.arange(0, 1 + resolution, resolution)
        xi, yi = interpolate.splev(x, tck, der=1)
        return xi, yi

    def set_nodes(self, *args):
        if len(args) == 1:
            x, y = zip(*args)
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError('Unrecognized node format')
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) == 0:
            self.x = list(x)
            self.y = list(y)
        else:
            m = np.isnan(x) | np.isnan(y)
            self.x = list(x[~m])
            self.y = list(y[~m])
        self.updated = True

    def add_node(self, x, y, hit_threshold=25):
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError('Point must be finite')
        if not self.has_node(x, y, hit_threshold):
            self.x.append(x)
            self.y.append(y)
            self.update_exclude()
            self.updated = True

    def has_node(self, x, y, hit_threshold):
        try:
            i = self.find_node(x, y, hit_threshold)
            return True
        except ValueError:
            return False

    def find_node(self, x, y, hit_threshold):
        xd = np.array(self.x) - x
        yd = np.array(self.y) - y
        d = np.sqrt(xd ** 2 + yd ** 2)
        i = np.argmin(d)
        if d[i] < hit_threshold:
            return i
        raise ValueError(f'No node within hit threshold of {hit_threshold}')

    def remove_node(self, x, y, hit_threshold=25):
        i = self.find_node(x, y, hit_threshold)
        self.x.pop(i)
        self.y.pop(i)
        self.update_exclude()
        self.updated = True

    def set_origin(self, x, y, hit_threshold=25):
        self.origin = int(self.find_node(x, y, hit_threshold))
        self.update_exclude()
        self.updated = True

    def nearest_point(self, x, y):
        xi, yi = self.interpolate()
        xd = np.array(xi) - x
        yd = np.array(yi) - y
        d = np.sqrt(xd ** 2 + yd ** 2)
        i = np.argmin(d)
        return xi[i], yi[i]

    def add_exclude(self, start, end):
        start = self.nearest_point(*start)
        end = self.nearest_point(*end)
        self.exclude.append((start, end))
        self.updated = True

    def update_exclude(self):
        new_exclude = []
        for s, e in self.exclude:
            try:
                s = self.nearest_point(*s)
                e = self.nearest_point(*e)
                if s == e:
                    continue
                new_exclude.append((s, e))
            except:
                pass
        self.exclude = new_exclude
        self.updated = True

    def remove_exclude(self, x, y):
        xi, yi = self.interpolate()
        pi = util.argnearest(x, y, xi, yi)
        for i, (s, e) in enumerate(self.exclude):
            si = util.argnearest(*s, xi, yi)
            ei = util.argnearest(*e, xi, yi)
            ilb, iub = min(si, ei), max(si, ei)
            if ilb <= pi <= iub:
                self.exclude.pop(i)
                self.updated = True
                break

    def simplify_exclude(self):
        xi, yi = self.interpolate()
        indices = []
        for s, e in self.exclude:
            si = util.argnearest(*s, xi, yi)
            ei = util.argnearest(*e, xi, yi)
            si, ei = min(si, ei), max(si, ei)
            indices.append([si, ei])

        indices = util.smooth_epochs(indices)
        self.exclude = [[[xi[si], yi[si]], [xi[ei], yi[ei]]] for si, ei in indices]
        self.updated = True

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "origin": self.origin,
            "exclude": self.exclude,
        }

    def set_state(self, state):
        x = np.array(state["x"])
        y = np.array(state["y"])
        m = np.isnan(x) | np.isnan(y)
        self.x = x[~m].tolist()
        self.y = y[~m].tolist()
        self.exclude = state.get("exclude", [])
        self.origin = state.get("origin", 0)
        self.updated = True


class Tile(Atom):

    info = Dict()
    image = Typed(np.ndarray)
    source = Typed(Path)
    extent = List()
    n_channels = Int()

    def __init__(self, info, image, source):
        self.info = info
        self.image = image
        self.source = source
        xlb, ylb, zlb = self.info["lower"][:3]

        # Images are in XYZC dimension. We need to calculate the upper extent
        # of the image so we can properly plot it.
        xpx, ypx, zpx = self.image.shape[:3]
        xv, yv, zv = self.info['voxel_size'][:3]
        xub = xlb + xpx * xv
        yub = ylb + ypx * yv
        zub = zlb + zpx * zv
        self.extent = [xlb, xub, ylb, yub, zlb, zub]
        self.n_channels = self.image.shape[-1]

    def contains(self, x, y):
        contains_x = self.extent[0] <= x <= self.extent[1]
        contains_y = self.extent[2] <= y <= self.extent[3]
        return contains_x and contains_y

    @classmethod
    def from_filename(cls, img_filename):
        image = np.load(img_filename)
        info = json.loads(img_filename.with_suffix('.json').read_text())
        return cls(info, image, img_filename)

    def to_coords(self, x, y, z=None):
        lower = self.info["lower"]
        voxel_size = self.info["voxel_size"]
        if z is None:
            indices = np.c_[x, y, np.full_like(x, lower[-1])]
        else:
            indices = np.c_[x, y, z]
        points = (indices * voxel_size) + lower
        if z is None:
            return points[:, :2].T
        return points.T

    def to_indices(self, x, y, z=None):
        lower = self.info["lower"]
        voxel_size = self.info["voxel_size"]
        if z is None:
            points = np.c_[x, y, np.full_like(x, lower[-1])]
        else:
            points = np.c_[x, y, z]
        indices = (points - lower) / voxel_size
        if z is None:
            return indices[:, :2].T
        return indices.T

    def to_indices_delta(self, v, axis='x'):
        if axis == 'x':
            return v / self.info['voxel_size'][0]
        elif axis == 'y':
            return v / self.info['voxel_size'][1]
        elif axis == 'z':
            return v / self.info['voxel_size'][2]
        else:
            raise ValueError('Unsupported axis')

    def nuclei_template(self, radius=2.5):
        voxel_size = self.info["voxel_size"][0]
        pixel_radius = int(np.round(radius / voxel_size))
        template = sphere(pixel_radius * 3, pixel_radius)
        return template / template.sum()

    def get_image_extent(self, axis='z'):
        x = self.extent[0:2]
        y = self.extent[2:4]
        z = self.extent[4:6]
        if axis == 'x':
            return tuple(y + z)
        if axis == 'y':
            return tuple(x + z)
        if axis == 'z':
            return tuple(x + y)

    def get_image(self, channel=None, z_slice=None, axis='z'):
        if z_slice is None:
            data = self.image.max(axis='xyz'.index(axis))
        else:
            data = self.image[:, :, z_slice, :]

        x, y = data.shape[:2]
        image = []
        for c, c_info in enumerate(self.info['channels']):
            if isinstance(channel, int):
                raise ValueError('Must provide name for channel')
            if channel is None or channel == 'All' or c_info['name'] == channel:
                color = c_info['display_color']
                rgb = colors.to_rgba(color)[:3]
                image.append(data[..., c][..., np.newaxis] * rgb)
        if len(image) == 0:
            raise ValueError(f'Channel {channel} does not exist')
        image = np.concatenate([i[np.newaxis] for i in image]).max(axis=0)
        return image / image.max(axis=(0, 1), keepdims=True)

    def get_state(self):
        return {"extent": self.extent}

    def set_state(self, state):
        self.extent = state["extent"]

    def map(self, x, y, channel, smooth_radius=2.5, width=5):
        """
        Calculate intensity in the specified channel for the xy coordinates.

        Optionally apply image smoothing and/or a maximum search.
        """
        # get_image returns a Nx3 array where the final dimension is RGB color.
        # We are only requesting one channel, but it is possible that the
        # information in the channel will be split among multiple RGB colors
        # depending on the specific color it is coded as. The sum should never
        # exceed 255.
        image = self.get_image(channel).sum(axis=-1)
        if smooth_radius:
            template = self.nuclei_template(smooth_radius)
            template = template.mean(axis=-1)
            image = signal.convolve2d(image, template, mode="same")

        if width:
            x, y = util.expand_path(x, y, width)

        xi, yi = self.to_indices(x.ravel(), y.ravel())
        i = ndimage.map_coordinates(image, [xi, yi])

        i.shape = x.shape
        if width is not None:
            i = i.max(axis=0)
        return i

    def center(self, dx, dy):
        '''
        Center tile origin with respect to dx and dy

        This is used for attempting to register images using phase cross-correlation
        '''
        extent = np.array(self.extent)
        width, height = extent[1:4:2] - extent[:4:2]
        self.extent = [dx, dx + width, dy, dy + height] + extent[4:]


class Piece:

    def __init__(self, tiles, path, piece):
        self.tiles = tiles
        self.path = path
        self.name = f'{path.stem}_piece_{piece}'
        self.piece = piece
        keys = 'IHC', 'OHC1', 'OHC2', 'OHC3', 'Extra'
        self.spirals = {k: Points() for k in keys}
        self.cells = {k: Points() for k in keys}

    @property
    def channel_names(self):
        # We assume that each tile has the same set of channels
        return [c['name'] for c in self.tiles[0].info['channels']]

    @classmethod
    def from_path(cls, path, piece=None):
        path = Path(path)
        tile_filenames = sorted(path.glob(f"*piece_{piece}*.npy"))
        log.info('Found tiles: %r', [t.stem for t in tile_filenames])
        tiles = [Tile.from_filename(f) for f in tile_filenames]

        # This pads the z-axis so that we have empty slices above/below stacks
        # such that they should align properly in z-space. This simplifies a
        # few downstream operations.
        slice_n = np.array([t.image.shape[2] for t in tiles])
        slice_lb = np.array([t.extent[4] for t in tiles])
        slice_ub = np.array([t.extent[5] for t in tiles])
        slice_scale = np.array([t.info['voxel_size'][2] for t in tiles])

        z_scale = slice_scale[0]
        z_min = min(slice_lb)
        z_max = max(slice_ub)
        z_n = int(np.ceil((z_max - z_min) / z_scale))

        pad_bottom = np.round((slice_lb - z_min) / z_scale).astype('i')
        pad_top = (z_n - pad_bottom - slice_n).astype('i')

        for (t, pb, pt) in zip(tiles, pad_bottom, pad_top):
            padding = [(0, 0), (0, 0), (pb, pt), (0, 0)]
            t.image = np.pad(t.image, padding)
            t.extent[4:] = [z_min, z_max]

        return cls(tiles, path, piece)

    def get_image_extent(self):
        extents = np.vstack([tile.get_image_extent() for tile in self.tiles])
        xmin = extents[:, 0].min()
        xmax = extents[:, 1].max()
        ymin = extents[:, 2].min()
        ymax = extents[:, 3].max()
        return [xmin, xmax, ymin, ymax]

    def merge_tiles(self):
        merged_lb = np.vstack([t.extent[::2] for t in self.tiles]).min(axis=0)
        merged_ub = np.vstack([t.extent[1::2] for t in self.tiles]).max(axis=0)
        voxel_size = self.tiles[0].info["voxel_size"]

        lb_pixels = np.floor(merged_lb / voxel_size).astype("i")
        ub_pixels = np.ceil(merged_ub / voxel_size).astype("i")
        extent_pixels = ub_pixels - lb_pixels
        shape = extent_pixels.tolist() + [self.tiles[0].n_channels]
        merged_image = np.full(shape, fill_value=0, dtype=int)

        for tile in self.tiles:
            tile_lb = tile.extent[::2]
            tile_lb = np.round((tile_lb - merged_lb) / voxel_size).astype("i")
            tile_ub = tile_lb + tile.image.shape[:-1]
            s = tuple(np.s_[lb:ub] for lb, ub in zip(tile_lb, tile_ub))
            merged_image[s] = tile.image

        info = {
            "lower": merged_lb,
            "voxel_size": voxel_size,
        }

        t_base = self.tiles[0]
        extra_keys = set(t_base.info.keys()) - set(('lower', 'voxel_size'))
        for k in extra_keys:
            for t in self.tiles[1:]:
                if t_base.info[k] != t.info[k]:
                    raise ValueError(f'Cannot merge tiles. {k} differs.')
            info[k] = t_base.info[k]
        return Tile(info, merged_image, self.path)

    def get_state(self):
        return {
            'tiles': {t.source.stem: t.get_state() for t in self.tiles},
            'spirals': {k: v.get_state() for k, v in self.spirals.items()},
            'cells': {k: v.get_state() for k, v in self.cells.items()},
        }

    def set_state(self, state):
        for k, v in self.spirals.items():
            v.set_state(state['spirals'][k])
        for k, v in self.cells.items():
            v.set_state(state['cells'][k])
        for tile in self.tiles:
            tile.set_state(state['tiles'][tile.source.stem])

    def guess_cells(self, cell_type, width, spacing, channel):
        log.info('Finding %s assuming within %f um of spiral and spaced %f microns on channel %s',
                 cell_type, width, spacing, channel)
        tile = self.merge_tiles()
        x, y = self.spirals[cell_type].interpolate(resolution=0.0001)
        i = tile.map(x, y, channel, width=width)
        xn, yn = util.find_nuclei(x, y, i, spacing=spacing)

        # Map to centroid
        xni, yni = tile.to_indices(xn, yn)
        image = tile.get_image(channel=channel).max(axis=-1)
        x_radius = tile.to_indices_delta(width, 'x')
        y_radius = tile.to_indices_delta(width, 'y')
        log.info('Searching for centroid within %ix%i pixels of spiral', x_radius,
                 y_radius)
        xnic, ynic = util.find_centroid(xni, yni, image, x_radius, y_radius, 4)
        xnc, ync = tile.to_coords(xnic, ynic)
        log.info('Shifted points up to %.0f x %.0f microns',
                 np.max(np.abs(xnc - xn)), np.max(np.abs(ync - yn)))
        self.cells[cell_type].set_nodes(xnc, ync)
        return len(xnc)

    def clear_cells(self, cell_type):
        self.cells[cell_type].set_nodes([], [])

    def clear_spiral(self, cell_type):
        self.spirals[cell_type].set_nodes([], [])


freq_fn = {
    'mouse': lambda d: (10**((1-d)*0.92) - 0.680) * 9.8,
}


class Cochlea:

    def __init__(self, pieces, path):
        self.pieces = pieces
        self.path = path
        self.name = path.stem

    @classmethod
    def from_path(cls, path):
        log.info('Loading cochlea from %s', path)
        path = Path(path)
        pieces = [Piece.from_path(path, p) for p in util.list_pieces(path)]
        return cls(pieces, path)

    def make_frequency_map(self, freq_start=4, freq_end=64, freq_step=0.5,
                           species='mouse', spiral='IHC'):
        # First, we need to merge the spirals
        xo, yo = 0, 0
        results = []
        for piece in self.pieces:
            s = piece.spirals[spiral]
            x, y = s.interpolate(resolution=0.001)
            if len(x) == 0:
                raise ValueError(f'Please check the {spiral} spiral on piece {piece.name} and try again.')
            x_norm = x - (x[0] - xo)
            y_norm = y - (y[0] - yo)
            xo = x_norm[-1]
            yo = y_norm[-1]
            i = np.arange(len(x)) / len(x)
            result = pd.DataFrame({
                'direction': s.direction(),
                'i': i,
                'x': x_norm,
                'y': y_norm,
                'x_orig': x,
                'y_orig': y,
                'piece': piece.piece,
            }).set_index(['piece', 'i'])
            results.append(result)
        results = pd.concat(results).reset_index()

        # Now we can do some distance calculations
        results['distance_mm'] = np.sqrt(results['x'].diff() ** 2 + results['y'].diff() ** 2).cumsum() * 1e-3
        results['distance_mm'] = results['distance_mm'].fillna(0)
        results['distance_norm'] = results['distance_mm'] / results['distance_mm'].max()
        results['frequency'] = freq_fn[species](results['distance_norm'])

        info = {}
        for freq in octave_space(freq_start, freq_end, freq_step):
            idx = (results['frequency'] - freq).abs().idxmin()
            info[freq] = results.loc[idx].to_dict()
        return info
