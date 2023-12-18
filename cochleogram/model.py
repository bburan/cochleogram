import logging
log = logging.getLogger(__name__)

from atom.api import Atom, Bool, Dict, Event, Int, List, Property, Str, Typed, Value
from matplotlib import colors
from matplotlib import transforms as T
import numpy as np
import pandas as pd

from psiaudio.util import octave_space
from scipy import interpolate
from scipy import ndimage
from scipy import signal
from skimage.registration import phase_cross_correlation
from skimage.color import rgb2gray

from raster_geometry import sphere

from ndimage_enaml.model import ChannelConfig, NDImage, NDImageCollection

from cochleogram import util
from cochleogram.config import CELLS, CHANNEL_CONFIG


class Points(Atom):

    x = List()
    y = List()
    origin = Int()
    exclude = List()
    labels = Dict()

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

    def get_labeled_nodes(self, label):
        coords = [coords for coords, labels in self.labels.items() if label in labels]
        if len(coords) == 0:
            return [(), ()]
        else:
            return list(zip(*coords))

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
        log.info('Removing node %d. Origin is %d.', i, self.origin)
        if self.origin > i:
            self.origin -= 1
        coords = self.x.pop(i), self.y.pop(i)
        if coords in self.labels:
            labels = self.labels.pop(coords)
            log.info('Removing label for node %d. Coords are %r. Labels were %r.', i, coords, labels)
        self.update_exclude()
        self.updated = True

    def label_node(self, x, y, label, toggle, hit_threshold=25):
        i = self.find_node(x, y, hit_threshold)
        coords = self.x[i], self.y[i]
        log.info('Labeling node %d as %s. Coords are %r.', i, label, coords)
        labels = self.labels.setdefault(coords, set())
        if label in labels:
            labels.remove(label)
        else:
            labels.add(label)
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

    def clear(self):
        self.exclude = []
        self.set_nodes([], [])

    def get_state(self):
        labels = list(self.labels.items())
        return {
            "x": self.x,
            "y": self.y,
            "origin": self.origin,
            "exclude": self.exclude,
            "labels": [[list(k), list(v)] for k, v in self.labels.items()],
        }

    def set_state(self, state):
        x = np.array(state["x"])
        y = np.array(state["y"])
        m = np.isnan(x) | np.isnan(y)
        self.labels = {tuple(k): set(v) for k, v in state.get("labels", [])}
        self.x = x[~m].tolist()
        self.y = y[~m].tolist()
        self.exclude = state.get("exclude", [])
        self.origin = state.get("origin", 0)
        self.updated = True


class Tile(NDImage):

    source = Str()

    def __init__(self, info, image, source):
        super().__init__(info, image)
        self.source = source


class CellAnalysis(NDImageCollection):

    spirals = Dict()
    cells = Dict()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spirals = {c: Points() for c in CELLS}
        self.cells = {c: Points() for c in CELLS}

    @property
    def channel_names(self):
        # We assume that each tile has the same set of channels
        return self.tiles[0].channel_names

    def guess_cells(self, cell_type, width, spacing, channel, z_slice):
        tile = self.merge_tiles()
        x, y = util.guess_cells(tile, self.spirals[cell_type], width, spacing,
                                channel, z_slice)
        self.cells[cell_type].set_nodes(x, y)
        return len(x)

    def clear_cells(self, cell_type):
        self.cells[cell_type].clear()

    def clear_spiral(self, cell_type):
        self.spirals[cell_type].clear()

    def get_state(self):
        return {
            'spirals': {k: v.get_state() for k, v in self.spirals.items()},
            'cells': {k: v.get_state() for k, v in self.cells.items()},
        }

    def set_state(self, state):
        for k, v in self.spirals.items():
            v.set_state(state['spirals'][k])
        for k, v in self.cells.items():
            v.set_state(state['cells'][k])


class Piece(CellAnalysis):

    piece = Value()
    copied_from = Str()
    region = Value()

    def __init__(self, tiles, piece, copied_from=None, region=None):
        super().__init__(tiles=tiles)
        self.piece = piece
        self.copied_from = copied_from
        self.region = region

    @property
    def is_copy(self):
        return bool(self.copied_from)

    def align_tiles(self, alignment_channel='MyosinVIIa'):
        # First, figure out the order in which we should work on the alignment.
        # Let's keep it basic by just sorting by lower left corner of the xy
        # coordinate.
        if len(self.tiles) < 2:
            return
        corners = [tuple(t.get_rotated_extent()[::2][:2]) for t in self.tiles]
        order = sorted(range(len(corners)), key=lambda x: corners[x])

        base_tile = self.tiles[order[0]]
        base_img = ndimage.rotate(base_tile.get_image(alignment_channel), base_tile.get_rotation())
        base_img = rgb2gray(base_img)
        base_mask = base_img > np.percentile(base_img, 95)

        x_um_per_px, y_um_per_px = base_tile.info['voxel_size'][:2]

        for i in order[1:]:
            tile = self.tiles[i]
            img = ndimage.rotate(tile.get_image(alignment_channel), tile.get_rotation())
            img = rgb2gray(img)
            mask = img > np.percentile(img, 95)
            result = phase_cross_correlation(base_img, img,
                                             reference_mask=base_mask,
                                             moving_mask=mask)
            x_shift, y_shift = result[0]
            extent = np.array(base_tile.extent[:])
            extent[0:2] += x_shift * x_um_per_px
            extent[2:4] += y_shift * y_um_per_px
            tile.extent = extent.tolist()
            base_tile = tile
            base_img = img
            base_mask = mask

    def get_state(self):
        state = super().get_state()
        state.update({
            'tiles': {t.source: t.get_state() for t in self.tiles},
            'copied_from': self.copied_from,
        })
        return state

    def set_state(self, state):
        super().set_state(state)
        for tile in self.tiles:
            tile.set_state(state['tiles'][tile.source])


freq_fn = {
    'mouse': lambda d: (10**((1-d)*0.92) - 0.680) * 9.8,
}


class Cochlea:

    def __init__(self, pieces):
        self.pieces = pieces
        self.pieces[0].region = 'hook'
        self.pieces[-1].region = 'apex'

    def __iter__(self):
        yield from self.pieces

    def __len__(self):
        return len(self.pieces)

    @property
    def channel_names(self):
        # We assume that each tile has the same set of channels
        names = set()
        for piece in self.pieces:
            names.update(piece.channel_names)
        return sorted(names)

    def get_image_extent(self):
        return self._get_extent(lambda p: p.get_image_extent())

    def get_rotated_extent(self):
        return self._get_extent(lambda p: p.get_rotated_extent())

    def _get_extent(self, cb):
        extents = np.vstack([cb(piece) for piece in self.pieces])
        xmin = extents[:, 0].min()
        xmax = extents[:, 1].max()
        ymin = extents[:, 2].min()
        ymax = extents[:, 3].max()
        return [xmin, xmax, ymin, ymax]

    def ihc_spiral_complete(self):
        for piece in self.pieces:
            s = piece.spirals['IHC']
            x, y = s.interpolate(resolution=0.001)
            if len(x) == 0:
                return False
        return True

    def make_frequency_map(self, freq_start=4, freq_end=64, freq_step=0.5,
                           species='mouse', spiral='IHC',
                           include_extremes=True):
        # First, we need to merge the spirals
        xo, yo = 0, 0
        results = []
        for piece in self.pieces:
            s = piece.spirals[spiral]
            x, y = s.interpolate(resolution=0.001)
            if len(x) == 0:
                raise ValueError(f'Please check the {spiral} spiral on piece {piece.piece} and try again.')
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

        if include_extremes:
            for ix in (0, -1):
                row = results.iloc[ix].to_dict()
                info[row['frequency']] = row

        return info


class TileAnalysis(CellAnalysis):

    name = Str()

    def __init__(self, tile, name):
        super().__init__(tiles=[tile])
        self.name = name


class TileAnalysisCollection:

    def __init__(self, tiles):
        self.tiles = tiles

    def __iter__(self):
        yield from self.tiles

    def __len__(self):
        return len(self.tiles)
