from matplotlib import pyplot as plt
from matplotlib import patheffects as path_effects
from matplotlib import transforms
import numpy as np


def _plot_piece(ax, piece, xo, yo, xmax, ymax, freq_map=None, cells=None,
                channels=None, label_piece=False, label_position='bottom'):
    tile = piece.merge_tiles()
    img = tile.get_image(channels=channels)
    extent = tile.get_image_extent()
    xr = extent[0] - xo
    yr = extent[2] - yo
    xe = extent[1] - extent[0]
    ye = extent[3] - extent[2]
    extent = (xo, xo+xe, yo, yo+ye)
    t = tile.get_image_transform() + ax.transData
    ax.imshow(img.swapaxes(0, 1), origin='lower', extent=extent, transform=t)

    if freq_map is not None:
        for freq, freq_df in freq_map.items():
            if freq_df['piece'] != piece.piece:
                continue
            x = freq_df['x_orig']
            y = freq_df['y_orig']
            f = f'{freq:.1f}'
            ax.plot(x-xr, y-yr, 'ko', mec='w', mew=2)
            t = ax.annotate(f, (x-xr, y-yr), (5, 5), color='white', textcoords='offset points')
            t.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal(),
            ])

    if cells is not None:
        for cell in cells:
            x, y = piece.cells[cell].get_nodes()
            x = np.subtract(x, xr)
            y = np.subtract(y, yr)
            ax.plot(x, y, 'w.')

    if label_piece:
        label = f'Piece {piece.piece}'
        if piece.region is not None:
            label = f'{label} ({piece.region})'
        if piece.copied_from:
            label = f'{label} from {piece.copied_from}'

        x = xo + xe / 2
        if label_position == 'bottom':
            y = yo
            va = 'top'
        else:
            y = yo + ye
            va = 'bottom'

        t = ax.text(x, y, label, color='black', ha='center', va=va)
        t.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),
            path_effects.Normal(),
        ])

    xo += xe
    ymax = max(ymax, yo+ye)
    xmax = max(xmax, xo)
    return xo, yo, xmax, ymax


def plot_piece(ax, piece):
    _plot_piece(ax, piece, 0, 0, 0, 0)


def plot_composite(cochlea, include_freq_map=True, cells=None, channels=None,
                   label_pieces=True):
    figure, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    if include_freq_map:
        freq_map = cochlea.make_frequency_map()
    else:
        freq_map = None

    xo, yo = 0, 0
    xmax, ymax = 0, 0
    for piece in cochlea.pieces[:3]:
        xo, yo, xmax, ymax = _plot_piece(ax, piece, xo, yo, xmax, ymax,
                                         freq_map, cells, channels,
                                         label_piece=label_pieces,
                                         label_position='bottom')
    xo, yo = 0, ymax
    for piece in cochlea.pieces[3:]:
        xo, yo, xmax, ymax = _plot_piece(ax, piece, xo, yo, xmax, ymax,
                                         freq_map, cells, channels,
                                         label_piece=label_pieces,
                                         label_position='top')
    ax.set_facecolor('k')
    ax.axis([0, xmax, 0, ymax])
    ax.set_xticks([])
    ax.set_yticks([])

    figure.subplots_adjust(left=0.025, right=0.975, top=0.9, bottom=0.025)
    return figure


def plot_slide_layout(cochlea):
    figure, ax = plt.subplots(1, 1)
    for piece in cochlea.pieces:
        for tile in piece.tiles:
            plot_tile(ax, tile)
    e = cochlea.get_rotated_extent()
    ax.axis(e)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('k')
    return figure


def plot_tile(ax, tile, **kwargs):
    ax.imshow(tile.get_image().swapaxes(0, 1),
              origin='lower',
              aspect='equal',
              transform=tile.get_image_transform() + ax.transData,
              extent=tile.get_image_extent(),
              **kwargs)
