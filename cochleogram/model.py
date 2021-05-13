from pathlib import Path

from atom.api import Atom, Dict, Event, Float, Int, List, Typed
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy import signal
from raster_geometry import sphere

from cochleogram import util


class Points(Atom):

    x = List()
    y = List()
    i = Int()
    updated = Event()

    def __init__(self, x=None, y=None, i=0):
        self.x = [] if x is None else x
        self.y = [] if y is None else y
        self.i = i

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
        i = self.i
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

    def interpolate(self, degree=3, smoothing=0, resolution=0.01):
        nodes = self.get_nodes()
        if len(nodes[0]) <= 3:
            return [], []
        tck, u = interpolate.splprep(nodes, k=degree, s=smoothing)
        x = np.arange(0, 1 + resolution, resolution)
        xi, yi = interpolate.splev(x, tck, der=0)
        return xi, yi

    def set_nodes(self, x, y):
        self.x = list(x)
        self.y = list(y)
        self.updated = True

    def add_node(self, x, y):
        if not self.has_node(x, y):
            self.x.append(x)
            self.y.append(y)
            self.updated = True

    def has_node(self, x, y, hit_threshold=1e-6):
        try:
            i = self.find_node(x, y, hit_threshold)
            return True
        except ValueError:
            return False

    def find_node(self, x, y, hit_threshold=1e-6):
        xd = np.array(self.x) - x
        yd = np.array(self.y) - y
        d = np.sqrt(xd ** 2 + yd ** 2)
        i = np.argmin(d)
        if d[i] < hit_threshold:
            return i
        raise ValueError('No node nearby')

    def remove_node(self, x, y, hit_threshold=1e-6):
        try:
            i = self.find_node(x, y, hit_threshold)
            self.x.pop(i)
            self.y.pop(i)
            self.updated = True
        except ValueError:
            pass

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "i": self.i,
        }

    def set_state(self, state):
        self.x = state["x"]
        self.y = state["y"]
        self.i = state["i"]
        self.updated = True


class Tile(Atom):

    info = Dict()
    image = Typed(np.ndarray)
    source = Typed(Path)
    extent = List()
    scaling = Float()

    def __init__(self, info, image, source):
        self.info = info
        self.image = image
        self.source = source
        ylb = self.info["lower"][1]
        yub = self.info["upper"][1]
        xlb = self.info["lower"][0]
        xub = self.info["upper"][0]
        self.extent = [xlb, xub, ylb, yub]
        self.scaling = self.info["scaling"][0]

    def contains(self, x, y):
        contains_x = self.extent[0] <= x <= self.extent[1]
        contains_y = self.extent[2] <= y <= self.extent[3]
        return contains_x and contains_y

    @classmethod
    def from_filename(cls, filename):
        info, image = util.load_data(filename)
        return cls(info, image, filename)

    def to_coords(self, x, y, z=None):
        lower = self.info["lower"]
        scaling = self.info["scaling"]
        if z is None:
            indices = np.c_[x, y, np.full_like(x, lower[-1])]
        else:
            indices = np.c_[x, y, z]
        points = (indices * scaling) + lower
        if z is None:
            return points[:, :2].T
        return points.T

    def to_indices(self, x, y, z=None):
        lower = self.info["lower"]
        scaling = self.info["scaling"]
        if z is None:
            points = np.c_[x, y, np.full_like(x, lower[-1])]
        else:
            points = np.c_[x, y, z]
        indices = (points - lower) / scaling
        if z is None:
            return indices[:, :2].T
        return indices.T

    def to_indices_delta(self, v, axis='x'):
        if axis == 'x':
            return v / self.info['scaling'][0]
        elif axis == 'y':
            return v / self.info['scaling'][1]
        elif axis == 'z':
            return v / self.info['scaling'][2]
        else:
            raise ValueError('Unsupported axis')

    def nuclei_template(self, radius=2.5e-6):
        scaling = self.info["scaling"][0]
        pixel_radius = int(np.round(radius / scaling))
        template = sphere(pixel_radius * 3, pixel_radius)
        return template / template.sum()

    def get_image(self, channel=None, z_slice=None, projection=True):
        if channel is None:
            channel = np.s_[:]
        if z_slice is None:
            z_slice = np.s_[:]
        image = self.image[:, :, z_slice, channel]
        if projection:
            image = image.max(axis=2)
        return image

    def get_state(self):
        return {"extent": self.extent}

    def set_state(self, state):
        self.extent = state["extent"]

    def map(self, x, y, channel, smooth_radius=2.5e-6, width=5e-6):
        """
        Calculate intensity in the specified channel for the xy coordinates.

        Optionally apply image smoothing and/or a maximum search.
        """
        image = self.get_image(channel)
        if smooth_radius:
            template = self.nuclei_template(smooth_radius)
            template = template.mean(axis=-1)
            image = signal.convolve2d(image, template, mode="same")

        if width:
            x, y = util.expand_path(x, y, width)

        xi, yi = self.to_indices(x.ravel(), y.ravel())
        i = ndimage.map_coordinates(image.T, [xi, yi])

        i.shape = x.shape
        if width is not None:
            i = i.max(axis=0)
        return i


class Piece:
    def __init__(self, tiles, path, piece):
        self.tiles = tiles
        self.path = path
        self.piece = piece
        keys = 'IHC', 'OHC1', 'OHC2', 'OHC3'
        self.spirals = {k: Points() for k in keys}
        self.cells = {k: Points() for k in keys}

    @classmethod
    def from_path(cls, path, piece):
        path = Path(path)
        tile_filenames = sorted(path.glob(f"*piece {piece}*"))
        tiles = [Tile.from_filename(f) for f in tile_filenames]

        # This pads the z-axis so that we have empty slices above/below stacks
        # such that they should align properly in z-space. This simplifies a
        # few downstream operations.
        slice_n = [t.image.shape[2] for t in tiles]
        slice_lb = [t.info['lower'][2] for t in tiles]
        slice_ub = [t.info['upper'][2] for t in tiles]
        slice_scale = [t.info['scaling'][2] for t in tiles]

        z_scale = slice_scale[0]
        z_min = min(slice_lb)
        z_max = max(slice_ub)
        z_n = int(np.ceil((z_max - z_min) / z_scale))

        pad_bottom = np.round((slice_lb - z_min) / z_scale).astype('i')
        pad_top = (z_n - pad_bottom - slice_n).astype('i')

        for (t, pb, pt) in zip(tiles, pad_bottom, pad_top):
            padding = [(0, 0), (0, 0), (pb, pt), (0, 0)]
            t.image = np.pad(t.image, padding)
            t.info['lower'][2] = z_min
            t.info['upper'][2] = z_max

        return cls(tiles, path, piece)

    def get_image_extent(self):
        extents = np.vstack([tile.extent for tile in self.tiles])
        xmin = extents[:, 0].min()
        xmax = extents[:, 1].max()
        ymin = extents[:, 2].min()
        ymax = extents[:, 3].max()
        return [xmin, xmax, ymin, ymax]

    def merge_tiles(self):
        merged_lb = np.vstack([tile.info["lower"] for tile in self.tiles]).min(axis=0)
        merged_ub = np.vstack([tile.info["upper"] for tile in self.tiles]).max(axis=0)
        scaling = self.tiles[0].info["scaling"]

        lb_pixels = np.floor(merged_lb / scaling).astype("i")
        ub_pixels = np.ceil(merged_ub / scaling).astype("i")
        extent_pixels = ub_pixels - lb_pixels
        shape = extent_pixels.tolist() + [3]
        merged_image = np.full(shape, fill_value=0, dtype=np.float)

        for tile in self.tiles:
            tile_lb = np.round((tile.info["lower"] - merged_lb) / scaling).astype("i")
            tile_ub = tile_lb + tile.image.shape[:-1]
            s = tuple(np.s_[lb:ub] for lb, ub in zip(tile_lb, tile_ub))
            merged_image[s] = tile.image[::-1, ::-1, ::-1].swapaxes(0, 1)

        info = {
            "lower": merged_lb,
            "upper": merged_ub,
            "extent": merged_ub - merged_lb,
            "scaling": scaling,
        }
        merged_image = merged_image.swapaxes(0, 1)
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

    def guess_cells(self, cell_type, width, spacing):
        channel = 0 if cell_type == 'IHC' else 1
        #width = 5e-6 if cell_type == 'IHC' else 2.5e-6
        tile = self.merge_tiles()
        x, y = self.spirals[cell_type].interpolate(resolution=0.0001)
        i = tile.map(x, y, channel, width=width)
        xn, yn = util.find_nuclei(x, y, i, spacing=spacing)

        # Map to centroid
        xni, yni = tile.to_indices(xn, yn)
        image = tile.get_image(channel=channel, projection=True)
        x_radius = tile.to_indices_delta(width, 'x')
        y_radius = tile.to_indices_delta(width, 'y')
        xnic, ynic = util.find_centroid(xni, yni, image, x_radius, y_radius, 4)
        xnc, ync = tile.to_coords(xnic, ynic)
        self.cells[cell_type].set_nodes(xnc, ync)
        return len(xnc)

    def clear_cells(self, cell_type):
        self.cells[cell_type].set_nodes([], [])

    def clear_spiral(self, cell_type):
        self.spirals[cell_type].set_nodes([], [])
