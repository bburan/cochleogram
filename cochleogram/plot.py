from matplotlib import pyplot as plt
from matplotlib import patheffects as path_effects


def _plot_piece(ax, piece, xo, yo, xmax, ymax, freq_map):
    tile = piece.merge_tiles()
    img = tile.get_image()
    extent = tile.get_image_extent()
    xr = extent[0] - xo
    yr = extent[2] - yo
    xe = extent[1] - extent[0]
    ye = extent[3] - extent[2]
    extent = (xo, xo+xe, yo, yo+ye)
    ax.imshow(img.swapaxes(0, 1), origin='lower', extent=extent)
    xo += xe
    ymax = max(ymax, yo+ye)
    xmax = max(xmax, xo)
    for freq, freq_df in freq_map.items():
        if freq_df['piece'] != piece.piece:
            continue
        x = freq_df['x_orig']
        y = freq_df['y_orig']
        f = f'{freq:.1f}'
        plt.plot(x-xr, y-yr, 'ko', mec='w', mew=2)
        t = plt.annotate(f, (x-xr, y-yr), (5, 5), color='white', textcoords='offset points')
        t.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal(),
        ])
    return xo, yo, xmax, ymax


def frequency_map(cochlea):
    figure, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    freq_map = cochlea.make_frequency_map()

    xo, yo = 0, 0
    xmax, ymax = 0, 0
    for piece in cochlea.pieces[:3]:
        xo, yo, xmax, ymax = _plot_piece(ax, piece, xo, yo, xmax, ymax, freq_map)
    xo, yo = 0, ymax
    for piece in cochlea.pieces[3:]:
        xo, yo, xmax, ymax = _plot_piece(ax, piece, xo, yo, xmax, ymax, freq_map)
    ax.set_facecolor('k')
    ax.axis([0, xmax, 0, ymax])
    ax.set_xticks([])
    ax.set_yticks([])

    figure.subplots_adjust(left=0.025, right=0.975, top=0.95, bottom=0.025)
    return figure


def plot_tile(ax, tile):
    ax.imshow(tile.get_image().swapaxes(0, 1),
              origin='lower',
              aspect='equal',
              transform=tile.get_image_transform() + ax.transData,
              extent=tile.get_image_extent())
