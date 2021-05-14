import re
from pathlib import Path
import pickle


from matplotlib import path as mpath
import numpy as np
import pandas as pd
from scipy import ndimage, signal

from aicspylibczi import CziFile


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


def make_plot_path(spline, regions):
    if len(regions) == 0:
        verts = np.zeros((0, 2))
        return mpath.Path(verts, [])

    path_data = []
    for s, e in regions:
        xs, ys = get_region(spline, s, e)
        xe, ye = expand_path(xs, ys, 15e-6)
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


def find_nuclei(x, y, i, spacing=5e-6):
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
        i = image[ylb:yub, xlb:xub]
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


def load_data(filename, max_xy=512, dtype='uint8', reload=False):
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
    info["origin"] = np.array([float(v) * 1e-6 for v in node.values()])
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
    p_piece = re.compile('.*piece (\d+)\w?')
    pieces = []
    for path in Path(path).glob('*piece *.*'):
        if path.name.endswith('.json'):
            continue
        piece = int(p_piece.match(path.stem).group(1))
        pieces.append(piece)
    return sorted(set(pieces))
