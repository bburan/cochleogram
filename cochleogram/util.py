from importlib.metadata import version
import json
import re
from pathlib import Path
import pickle

from matplotlib import path as mpath
import numpy as np
import pandas as pd
from scipy import ndimage, optimize, signal



def get_region(spline, start, end):
    x1, y1 = start
    x2, y2 = end
    xi, yi = spline.interpolate(resolution=0.001)
    i1 = argnearest(x1, y1, xi, yi)
    i2 = argnearest(x2, y2, xi, yi)
    ilb = min(i1, i2)
    iub = max(i1, i2)
    xs, ys = xi[ilb:iub], yi[ilb:iub]
    return xs, ys


def make_plot_path(spline, regions, path_width=15):
    if len(regions) == 0:
        verts = np.zeros((0, 2))
        return mpath.Path(verts, [])

    path_data = []
    for s, e in regions:
        xs, ys = get_region(spline, s, e)
        xe, ye = expand_path(xs, ys, 15)
        xlb, xub = xe[0, :], xe[-1, :]
        ylb, yub = ye[0, :], ye[-1, :]
        xc = np.r_[xlb[1:], xub[::-1]]
        yc = np.r_[ylb[1:], yub[::-1]]
        path_data.append((mpath.Path.MOVETO, [xlb[0], ylb[0]]))
        for x, y in zip(xc, yc):
            path_data.append((mpath.Path.LINETO, (x, y)))
        path_data.append((mpath.Path.CLOSEPOLY, [xlb[0], ylb[0]]))
    codes, verts = zip(*path_data)
    return mpath.Path(verts, codes)


def argnearest(x, y, xa, ya):
    xd = np.array(xa) - x
    yd = np.array(ya) - y
    d = np.sqrt(xd ** 2 + yd ** 2)
    return np.argmin(d)


def expand_path(x, y, width):
    v = x + y * 1j
    a = np.angle(np.diff(v)) + np.pi / 2
    a = np.pad(a, (1, 0), mode='edge')
    dx = width * np.cos(a)
    dy = width * np.sin(a)
    x = np.linspace(x - dx, x + dx, 100)
    y = np.linspace(y - dy, y + dy, 100)
    return x, y


def find_nuclei(x, y, i, spacing=5):
    xy_delta = np.mean(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    distance = np.floor(spacing / xy_delta)
    p, _ = signal.find_peaks(i, distance=distance)
    return x[p], y[p]


def find_centroid(x, y, image, rx, ry, factor=4):
    x_center, y_center = [], []
    x = np.asarray(x)
    y = np.asarray(y)
    for xi, yi in zip(x, y):
        ylb, yub = int(round(yi-ry)), int(round(yi+ry))
        xlb, xub = int(round(xi-rx)), int(round(xi+rx))
        i = image[ylb:yub, xlb:xub].astype('int64')
        yc, xc = ndimage.center_of_mass(i ** factor)
        x_center.append(xc - rx)
        y_center.append(yc - ry)
    x_center = x + np.array(x_center)
    y_center = y + np.array(y_center)
    return x_center, y_center


def shortest_path(x, y, i=0):
    """
    Simple algorithm that assumes that the next "nearest" node is the one we
    want to draw a path through. This avoids trying to solve the complete
    traveling salesman problem.
    """
    # TODO: just use method in model
    nodes = list(zip(x, y))
    path = []
    while len(nodes) > 1:
        n = nodes.pop(i)
        path.append(n)
        d = np.sqrt(np.sum((np.array(nodes) - n) ** 2, axis=1))
        i = np.argmin(d)
    path.extend(nodes)
    return list(zip(*path))


def list_lif_stacks(filename):
    from readlif.reader import LifFile
    fh = LifFile(filename)
    return [stack.name for stack in fh.get_iter_image()]


def load_lif(filename, piece, max_xy=512, dtype='uint8', reprocess=False):
    filename = Path(filename)
    cache_filename = (
        filename.parent
        / filename.stem
        / (filename.stem + f'_{piece}')
    )
    info_filename = cache_filename.with_suffix('.json')
    img_filename = cache_filename.with_suffix('.npy')
    if not reprocess and info_filename.exists() and img_filename.exists():
        info = json.loads(info_filename.read_text())
        img = np.load(img_filename)
        return info, img

    from readlif.reader import LifFile
    from readlif.utilities import get_xml
    fh = LifFile(filename)
    for stack in fh.get_iter_image():
        if stack.name == piece:
            break
    else:
        raise ValueError(f'{piece} not found in {filename}')

    root, _ = get_xml(filename)
    node = root.find(f'.//Element[@Name="{piece}"]')

    y_pos = float(node.find('.//FilterSettingRecord[@Attribute="XPos"]').attrib['Variant'])
    x_pos = float(node.find('.//FilterSettingRecord[@Attribute="YPos"]').attrib['Variant'])
    # This seems to work for the Z-axis.
    z_pos = float(node.find('.//DimensionDescription[@DimID="3"]').attrib['Origin'])

    system_number = node.find('.//FilterSettingRecord[@Attribute="System_Number"]').attrib['Variant']
    system_type = node.find('.//ScannerSettingRecord[@Identifier="SystemType"]').attrib['Variant']
    system = f'{system_type} {system_number}'

    pixels = np.array(stack.dims[:3])
    voxel_size = 1 / np.array(stack.scale[:3])
    lower = np.array([x_pos, y_pos, z_pos]) * 1e6

    zoom = max_xy / max(pixels[:2])
    voxel_size[:2] /= zoom

    shape = [max_xy, max_xy, stack.dims[2], stack.channels]
    img = np.empty(shape, dtype=np.float32)
    for c in range(stack.channels):
        for z, s in enumerate(stack.get_iter_z(c=c)):
            img[:, :, z, c] = ndimage.zoom(s, (zoom, zoom))

    # Z-step was negative. Flip stack to fix this so that we always have a
    # positive Z-step.
    if voxel_size[2] < 0:
        img = img[:, :, ::-1]
        voxel_size[2] = -voxel_size[2]

    # Note that all units should be in microns since this is the most logical
    # unit for a confocal analysis.
    info = {
        # XYZ voxel size in microns (um).
        'voxel_size': voxel_size.tolist(),
        # XYZ origin in microns (um).
        'lower': lower.tolist(),
        # Store version number of cochleogram along with 
        'version': version('cochleogram'),
        # Reader used to read in data
        'reader': 'lif',
        # System used. I am including this information just in case we have to
        # implement specific tweaks for each confocal system we use.
        'system': system,
        'note': 'XY position from stage coords seem to be swapped',
        'channels': [
            {'name': 'CtBP2', 'display_color': 'red'},
            {'name': 'MyosinVIIa', 'display_color': 'blue'},
            {'name': 'PMT', 'display_color': 'white'},
        ]
    }

    # Rescale to range 0 ... 1
    img = img / img.max(axis=(0, 1, 2), keepdims=True)
    if 'int' in dtype:
        img *= 255

    # Coerce to dtype, reorder so that tile origin is in lower corner of image
    # (makes it easer to reconcile with plotting), and swap axes from YX to XY.
    # Final axes ordering should be XYZC where C is channel and origin of XY
    # should be in lower corner of screen.
    img = img.astype(dtype)[::-1].swapaxes(0, 1)

    cache_filename.parent.mkdir(exist_ok=True, parents=True)
    info_filename.write_text(json.dumps(info, indent=2))
    np.save(img_filename, img, allow_pickle=False)
    return info, img


def process_lif(filename, reprocess, cb=None):
    pieces = list_lif_stacks(filename)
    n_pieces = len(pieces)
    if cb is None:
        cb = lambda x: x
    for p, piece in enumerate(pieces):
        _ = load_lif(filename, piece, reprocess=reprocess)
        progress = int((p + 1) / n_pieces * 100)
        cb(progress)


def load_czi(filename, max_xy=512, dtype='uint8', reload=False):
    raise NotImplementedError
    # Note, this needs to be updated since I made some modifications to support
    # Leica LIF format and that includes changing imshow origin from lower to
    # upper since I was using the Leica viewer to make sure I got the extents
    # aligned properly.

    filename = Path(filename)
    cache_filename = (
        filename.parent
        / "processed"
        / f"max_xy_{max_xy}_dtype_{dtype}"
        / filename.with_suffix(".pkl").name
    )
    if not reload and cache_filename.exists():
        with cache_filename.open("rb") as fh:
            return pickle.load(fh)

    from aicspylibczi import CziFile
    fh = CziFile(filename)

    x_pixels = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/DimensionX"
        ).text
    )
    y_pixels = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/DimensionY"
        ).text
    )
    z_pixels = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/DimensionZ"
        ).text
    )

    x_scaling = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/ScalingX"
        ).text
    )
    y_scaling = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/ScalingY"
        ).text
    )
    z_scaling = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/ScalingZ"
        ).text
    )

    x_offset = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/OffsetX"
        ).text
    )
    y_offset = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/OffsetY"
        ).text
    )
    z_offset = float(
        fh.meta.find(
            "Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/AcquisitionModeSetup/OffsetZ"
        ).text
    )

    node = fh.meta.find(
        "Metadata/Information/Image/Dimensions/S/Scenes/Scene/Positions/Position"
    )

    info = {}
    info["offset"] = np.array([x_offset, y_offset, z_offset])
    info["pixels"] = np.array([x_pixels, y_pixels, z_pixels]).astype("i")
    info["scaling"] = np.array([x_scaling, y_scaling, z_scaling])
    info["origin"] = np.array([float(v) * 1e-6 for k, v in node.items()])
    info["lower"] = info["origin"]
    info["extent"] = info["pixels"] * info["scaling"]
    info["upper"] = info["lower"] + info["extent"]
    del info["pixels"]

    img = fh.read_image()[0][0, 0, 0]

    # First, do the zoom. This is the best time to handle it before we do
    # additional manipulations.
    zoom = max_xy / max(x_pixels, y_pixels)
    if zoom < 1:
        img = np.concatenate([ndimage.zoom(i, (1, zoom, zoom))[np.newaxis] for i in img])
        info["scaling"][:2] /= zoom

    # Initial ordering is czyx
    #                     0123
    # Final ordering      xyzc
    img = img.swapaxes(0, 3).swapaxes(1, 2)

    # Add a third channel to allow for RGB images
    padding = [(0, 0)] * img.ndim
    padding[-1] = (0, 1)
    img = np.pad(img, padding, "constant")

    # Rescale to range 0 ... 1
    img = img / img.max(axis=(0, 1, 2), keepdims=True)
    if 'int' in dtype:
        img *= 255
    img = img.astype(dtype)

    cache_filename.parent.mkdir(exist_ok=True, parents=True)
    with cache_filename.open("wb") as fh:
        pickle.dump((info, img), fh, pickle.HIGHEST_PROTOCOL)

    return info, img


def list_pieces(path):
    p_piece = re.compile('.*piece_(\d+)\w?')
    pieces = []
    for path in Path(path).glob('*piece_*.*'):
        if path.name.endswith('.json'):
            continue
        piece = int(p_piece.match(path.stem).group(1))
        pieces.append(piece)
    return sorted(set(pieces))


def smooth_epochs(epochs):
    '''
    Given a 2D array of epochs in the format [[start time, end time], ...],
    identify and remove all overlapping epochs such that::
        [ epoch   ]        [ epoch ]
            [ epoch ]
    Will become::
        [ epoch     ]      [ epoch ]
    Epochs do not need to be ordered when provided; however, they will be
    returned ordered.
    '''
    if len(epochs) == 0:
        return epochs
    epochs = np.asarray(epochs)
    epochs.sort(axis=0)
    i = 0
    n = len(epochs)
    smoothed = []
    while i < n:
        lb, ub = epochs[i]
        i += 1
        while (i < n) and (ub >= epochs[i,0]):
            ub = epochs[i,1]
            i += 1
        smoothed.append((lb, ub))
    return np.array(smoothed)


def arc_origin(x, y):
    '''
    Determine most likely origin for arc
    '''
    def _fn(origin, xa, ya):
        xo, yo = origin
        d = np.sqrt((xa - xo) ** 2 + (ya - yo) ** 2)
        return np.sum(np.abs(d - d.mean()))
    result = optimize.minimize(_fn, (x.mean(), y.mean()), (x, y))
    return result.x


def arc_direction(x, y):
    '''
    Given arc defined by x and y, determine direction of arc

    Parameters
    ----------
    x : array
        x coordinates of vertices defining arc
    y : array
        y coordinates of vertices defining arc

    Returns
    -------
    direction : int
        -1 if arc sweeps clockwise (i.e., change in angle of vertices relative
        to origin is negative), +1 if arc sweeps counter-clockwise
    '''
    xo, yo = arc_origin(x, y)
    angles = np.unwrap(np.arctan2(y-yo, x-xo))
    sign = np.sign(np.diff(angles))
    if np.any(sign != sign[0]):
        raise ValueError('Cannot determine direction of arc')
    return sign[0]
