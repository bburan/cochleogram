import argparse
from pathlib import Path

import enaml
from enaml.qt.qt_application import QtApplication

from cochleogram.model import Piece
from cochleogram.presenter import Presenter
from cochleogram.util import list_pieces

with enaml.imports():
    from cochleogram.gui import CochleagramWindow


def main_prepare():
    parser = argparse.ArgumentParser('Create cached files for cochleogram')
    parser.add_argument('path')

    args = parser.parse_args()
    src = Path(args.path)
    for piece in list_pieces(src / 'cochleograms'):
        # We don't need to do anything. Just create the cache.
        print(f'Processing piece {piece}')
        Piece.from_path(src / 'cochleograms', piece)

    src_cache = src / 'cochleograms' / 'processed' / 'max_xy_512_dtype_uint8'
    dest = src / 'cochleogram_analysis'
    src_cache.rename(dest)


def main():
    parser = argparse.ArgumentParser("Cochleogram helper")
    parser.add_argument("path", nargs='?')
    parser.add_argument("--piece")
    args = parser.parse_args()

    app = QtApplication()

    if args.path is not None:
        pieces = list_pieces(path) if args.piece is None else [args.piece]
        presenters = [Presenter(Piece.from_path(args.path, p)) for p in pieces]
    else:
        presenters = []

    view = CochleagramWindow(presenters=presenters)
    view.show()
    app.start()
    app.stop()


if __name__ == "__main__":
    main()
