import argparse
import configparser
import logging
from pathlib import Path

from enaml.application import deferred_call
from enaml.qt.QtCore import QStandardPaths


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


def main_prepare_lif():
    from cochleogram.util import process_lif
    parser = argparse.ArgumentParser('Create cached files for cochleogram from LIF files')
    parser.add_argument('path')
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()
    filename = Path(args.path)
    process_lif(filename, args.reprocess)


def main():
    import enaml
    from enaml.qt.qt_application import QtApplication
    logging.basicConfig(level='INFO')

    with enaml.imports():
        from cochleogram.gui import CochleagramWindow, load_dataset

    parser = argparse.ArgumentParser("Cochleogram helper")
    parser.add_argument("path", nargs='?')
    args = parser.parse_args()

    app = QtApplication()
    config = get_config()

    current_path = config['DEFAULT']['current_path']
    view = CochleagramWindow(current_path=current_path)
    if args.path is not None:
        deferred_call(load_dataset, args.path, view)
    view.show()
    app.start()
    app.stop()
    config['DEFAULT']['current_path'] = str(Path(view.current_path).absolute())
    write_config(config)


if __name__ == "__main__":
    main()
