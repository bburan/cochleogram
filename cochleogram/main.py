from pathlib import Path
import re

import enaml
from enaml.qt.qt_application import QtApplication

from cochleogram.model import Piece
from cochleogram.presenter import Presenter

with enaml.imports():
    from cochleogram.gui import CochleagramWindow


def launcher():


def main():
    import argparse

    parser = argparse.ArgumentParser("Cochleogram helper")
    parser.add_argument("path")
    parser.add_argument("--piece")
    args = parser.parse_args()

    if args.piece is None:
        p_piece = re.compile('.*piece (\d+)\w?')
        pieces = []
        for path in Path(args.path).glob('*piece*.czi'):
            piece = int(p_piece.match(path.stem).group(1))
            pieces.append(piece)
        pieces = sorted(set(pieces))
    else:
        pieces = [args.piece]

    presenters = [Presenter(Piece.from_path(args.path, p)) for p in pieces]
    app = QtApplication()
    view = CochleagramWindow(presenters=presenters)
    view.show()
    app.start()
    app.stop()


if __name__ == "__main__":
    main()
