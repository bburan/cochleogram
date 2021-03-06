import argparse
import configparser
from pathlib import Path

from enaml.qt.QtCore import QStandardPaths

from cochleogram.model import Piece
from cochleogram.util import list_pieces, load_data


def config_file():
    config_path = Path(QStandardPaths.standardLocations(QStandardPaths.AppConfigLocation)[0])
    config_file =  config_path / 'cochleogram' / 'config.ini'
    config_file.parent.mkdir(exist_ok=True, parents=True)
    return config_file


def get_config():
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'current_path': ''}
    config.read(config_file())
    return config


def write_config(config):
    with config_file().open('w') as fh:
        config.write(fh)


def main_prepare():
    parser = argparse.ArgumentParser('Create cached files for cochleogram')
    parser.add_argument('path')

    args = parser.parse_args()
    src = Path(args.path)
    cochleogram_path = src / 'cochleograms'
    for filename in cochleogram_path.glob('*.czi'):
        # We don't need to do anything. Just create the cache which occurs when
        # we load data for the first time.
        print(f'Processing file {filename}')
        _ = load_data(filename)

    src_cache = src / 'cochleograms' / 'processed' / 'max_xy_512_dtype_uint8'
    dest = src / 'cochleogram_analysis'
    src_cache.rename(dest)


def main():
    import enaml
    from enaml.qt.qt_application import QtApplication

    from cochleogram.presenter import Presenter

    with enaml.imports():
        from cochleogram.gui import CochleagramWindow

    parser = argparse.ArgumentParser("Cochleogram helper")
    parser.add_argument("path", nargs='?')
    parser.add_argument("--piece")
    args = parser.parse_args()

    app = QtApplication()
    config = get_config()

    if args.path is not None:
        pieces = list_pieces(args.path) if args.piece is None else [args.piece]
        presenters = [Presenter(Piece.from_path(args.path, p)) for p in pieces]
    else:
        presenters = []

    current_path = config['DEFAULT']['current_path']
    view = CochleagramWindow(presenters=presenters, current_path=current_path)
    view.show()
    app.start()
    app.stop()
    config['DEFAULT']['current_path'] = str(Path(view.current_path).absolute())
    write_config(config)


if __name__ == "__main__":
    main()
