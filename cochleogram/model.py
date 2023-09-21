import logging
log = logging.getLogger(__name__)

from atom.api import Atom, Bool, Dict, Event, Float, Int, List, Str, Typed
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

from cochleogram import util
from cochleogram.config import CELLS, CHANNEL_CONFIG


class ChannelConfig(Atom):

    name = Str()
    min_value = Float(0)
    max_value = Float(1)
    visible = Bool(True)


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
        log.info('Removing node %d. Origin is %d.', i, self.origin)
        if self.origin > i:
            self.origin -= 1
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

    def clear(self):
        self.exclude = []
        self.set_nodes([], [])

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
    source = Str()
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

    @property
    def channel_names(self):
        return [c['name'] for c in self.info['channels']]

    def contains(self, x, y):
        contains_x = self.extent[0] <= x <= self.extent[1]
        contains_y = self.extent[2] <= y <= self.extent[3]
        return contains_x and contains_y

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

    def get_image_extent(self, axis='z', norm=False):
        e = np.array(self.extent).reshape((3, 2))
        if norm:
            e = e - e[:, [0]]
        extent = e.ravel().tolist()
        x = extent[0:2]
        y = extent[2:4]
        z = extent[4:6]
        if axis == 'x':
            return tuple(y + z)
        if axis == 'y':
            return tuple(x + z)
        if axis == 'z':
            return tuple(x + y)

    def get_image_transform(self):
        return T.Affine2D().rotate_deg_around(*self.get_image_center(),
                                              self.get_rotation())

    def get_rotated_extent(self):
        '''
        Calculate the new extents of the tile after rotation.

        This assumes that the tile is rotated using scipy.ndimage where the
        resulting array is reshaped to ensure that the input image is contained
        entirely in the output image.
        '''
        e = self.extent[:]
        ll = e[0], e[2]
        lr = e[1], e[2]
        ul = e[0], e[3]
        ur = e[1], e[3]
        coords = np.array([ll, lr, ur, ul, ll])
        t_coords = self.get_image_transform().transform(coords)
        xlb, ylb = t_coords.min(axis=0)
        xub, yub = t_coords.max(axis=0)
        e[:4] = xlb, xub, ylb, yub
        return e

    def get_image_center(self, axis='z', norm=False):
        extent = self.get_image_extent()
        center = np.array(extent).reshape((2, 2)).mean(axis=1)
        return tuple(center)

    def get_rotation(self):
        return self.info.get('rotation', 0)

    def get_image(self, channels=None, z_slice=None, axis='z',
                  norm_percentile=99):
        if z_slice is None:
            data = self.image.max(axis='xyz'.index(axis))
        else:
            data = self.image[:, :, z_slice, :]
        x, y = data.shape[:2]

        # Normalize data
        data_max =  np.percentile(data, norm_percentile, axis=(0, 1), keepdims=True)
        data_mask = data_max != 0
        data = np.divide(data, data_max, where=data_mask).clip(0, 1)

        if channels is None:
            channels = self.channel_names
        elif isinstance(channels, int):
            raise ValueError('Must provide name for channel')
        elif isinstance(channels, str):
            channels = [channels]
        elif len(channels) == 0:
            #raise ValueError('Cannot generate image with zero channels')
            return np.zeros((x, y))

        # Check that channels are valid and generate config
        channel_config = {}
        for c in channels:
            if isinstance(c, ChannelConfig):
                if not c.visible:
                    continue
                if c.name not in self.channel_names:
                    raise ValueError(f'Channel {c.name} does not exist')
                channel_config[c.name] = {
                    'min_value': c.min_value,
                    'max_value': c.max_value,
                    **CHANNEL_CONFIG[c.name],
                }
            elif isinstance(c, dict):
                channel_config[c['name']] = {
                    **c,
                    **CHANNEL_CONFIG[c['name']],
                }
            elif c not in self.channel_names:
                raise ValueError(f'Channel {c} does not exist')
            else:
                channel_config[c] = {
                    'min_value': 0,
                    'max_value': 1,
                    **CHANNEL_CONFIG[c],
                }

        image = []
        for c, c_info in enumerate(self.info['channels']):
            if c_info['name'] in channel_config:
                config = channel_config[c_info['name']]
                rgb = colors.to_rgba(config['display_color'])[:3]

                lb = config['min_value']
                ub = config['max_value']
                d = np.clip((data[..., c] - lb) / (ub - lb), 0, 1)
                d = d[..., np.newaxis] * rgb
                image.append(d)

        return np.concatenate([i[np.newaxis] for i in image]).max(axis=0)

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


class CellAnalysis:

    def __init__(self):
        self.spirals = {c: Points() for c in CELLS}
        self.cells = {c: Points() for c in CELLS}

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

    def __init__(self, tiles, piece, copied_from=None, region=None):
        super().__init__()
        self.tiles = tiles
        self.piece = piece
        self.copied_from = copied_from
        self.region = region

    def __iter__(self):
        yield from self.tiles

    @property
    def is_copy(self):
        return bool(self.copied_from)

    @property
    def channel_names(self):
        # We assume that each tile has the same set of channels
        return self.tiles[0].channel_names

    def get_image_extent(self):
        return self._get_extent(lambda t: t.get_image_extent())

    def get_rotated_extent(self):
        return self._get_extent(lambda t: t.get_rotated_extent())

    def _get_extent(self, cb):
        extents = np.vstack([cb(tile) for tile in self.tiles])
        xmin = extents[:, 0].min()
        xmax = extents[:, 1].max()
        ymin = extents[:, 2].min()
        ymax = extents[:, 3].max()
        return [xmin, xmax, ymin, ymax]

    def merge_tiles(self, flatten=True):
        '''
        Merges the information from the tiles into one single tile representing the piece

        This is typically used when we need to do analyses that function across
        the individual tiles.
        '''
        merged_lb = np.vstack([t.get_rotated_extent()[::2] for t in self.tiles]).min(axis=0)
        merged_ub = np.vstack([t.get_rotated_extent()[1::2] for t in self.tiles]).max(axis=0)
        voxel_size = self.tiles[0].info["voxel_size"]
        lb_pixels = np.floor(merged_lb / voxel_size).astype("i")
        ub_pixels = np.ceil(merged_ub / voxel_size).astype("i")
        extent_pixels = ub_pixels - lb_pixels
        shape = extent_pixels.tolist() + [self.tiles[0].n_channels]
        merged_image = np.full(shape, fill_value=0, dtype=int)
        merged_n = np.full(shape, fill_value=0, dtype=int)

        for i, tile in enumerate(self.tiles):
            if flatten:
                img = tile.image.max(axis=2, keepdims=True)
            else:
                img = tile.image

            if tile.get_rotation() != 0:
                img = ndimage.rotate(img, tile.get_rotation(), cval=np.nan, order=0)

            tile_lb = tile.get_rotated_extent()[::2]
            tile_lb = np.round((tile_lb - merged_lb) / voxel_size).astype("i")
            tile_ub = tile_lb + img.shape[:-1]
            s = tuple([np.s_[lb:ub] for lb, ub in zip(tile_lb, tile_ub)])
            merged_image[s] += img
            merged_n[s] += 1

        merged_image = merged_image / merged_n
        merged_image = merged_image.astype('i')

        info = {
            "lower": merged_lb,
            "voxel_size": voxel_size,
            "rotation": 0,
        }

        t_base = self.tiles[0]
        extra_keys = set(t_base.info.keys()) - set(('lower', 'voxel_size', 'rotation'))
        for k in extra_keys:
            for t in self.tiles[1:]:
                if t_base.info[k] != t.info[k]:
                    raise ValueError(f'Cannot merge tiles. {k} differs.')
            info[k] = t_base.info[k]
        return Tile(info, merged_image, f'piece_{self.piece}_merged')

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
            x_shift, y_shift = phase_cross_correlation(base_img, img,
                                                       reference_mask=base_mask,
                                                       moving_mask=mask)
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

    def __init__(self, tile, name):
        super().__init__()
        self.tile = tile
        self.name = name

    def merge_tiles(self):
        return self.tile

    def get_image_extent(self):
        return self.tile.get_image_extent()

    @property
    def channel_names(self):
        # We assume that each tile has the same set of channels
        return self.tile.channel_names

    def __iter__(self):
        yield from [self.tile]


class TileAnalysisCollection:

    def __init__(self, tiles):
        self.tiles = tiles

    def __iter__(self):
        yield from self.tiles
