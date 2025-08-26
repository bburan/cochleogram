import logging

log = logging.getLogger(__name__)

import json
from pathlib import Path
import re

import numpy as np

from . import model
from . import util

P_FREQ = re.compile(r'.*?(\d+p\d+)_kHz.*')


def extract_frequency(name):
    if (m := P_FREQ.match(name)) is not None:
        return float(m.group(1).replace('p', '.'))
    return None


class BaseReader:
    '''
    Base class of all readers. Provides state persistence for analysis. Actual
    loading of data and generation of filenames for saving state persistence
    data should be handled by subclasses.
    '''

    def load_state(self, obj):
        state_filename = self.state_filename(obj)
        if not state_filename.exists():
            raise IOError('No saved analysis found')
        return json.loads(state_filename.read_text())

    def save_state(self, obj, state):
        state_filename = self.state_filename(obj)
        state_filename.parent.mkdir(exist_ok=True)
        state_filename.write_text(json.dumps(state, indent=4))

    def load_collection(self, load_analysis=True,
                     raise_load_analysis_error=False):
        collection = self._load_collection()
        if load_analysis:
            for obj in collection:
                try:
                    state = self.load_state(obj)
                    obj.set_state(state['data'])
                except IOError:
                    if raise_load_analysis_error:
                        raise
        return collection

    def _load_collection(self):
        raise NotImplementedError

    def state_filename(self, obj):
        raise NotImplementedError

    def get_name(self):
        return self.path.stem


class CochleaReader(BaseReader):
    '''
    Reads an entire cochlea

    This expects multiple images with at least one image per piece, possibly
    more if the entire piece did not fit inside the field of view.
    '''

    def __init__(self, path):
        self.path = Path(path)

    def save_figure(self, fig, suffix, file_format='pdf'):
        filename = self.save_path() / f'{self.get_name()}_{suffix}.{file_format}'
        filename.parent.mkdir(exist_ok=True)
        fig.savefig(filename)

    def state_filename(self, piece):
        return self.save_path() / f'{self.path.stem}_piece_{piece.piece}_analysis.json'

    def load_piece(self, piece, stack_names):
        tiles = [self._load_tile(sn) for sn in stack_names]

        copy = re.compile(fr'^piece_{piece}\w?_copied_([\w-]+)')
        copied = set()
        for sn in stack_names:
            if (m := copy.match(sn)) is not None:
                copied.add(m.group(1))
        copied = ', '.join(sorted(copied))

        # This pads the z-axis so that we have empty slices above/below stacks
        # such that they should align properly in z-space. This simplifies a
        # few downstream operations.
        slice_n = np.array([t.image.shape[2] for t in tiles])
        slice_lb = np.array([t.extent[4] for t in tiles])
        slice_ub = np.array([t.extent[5] for t in tiles])
        slice_scale = np.array([t.info['voxel_size'][2] for t in tiles])

        z_scale = slice_scale[0]
        z_min = min(slice_lb)
        z_max = max(slice_ub)
        z_n = int(np.ceil((z_max - z_min) / z_scale))

        pad_bottom = np.round((slice_lb - z_min) / z_scale).astype('i')
        pad_top = (z_n - pad_bottom - slice_n).astype('i')

        for (t, pb, pt) in zip(tiles, pad_bottom, pad_top):
            padding = [(0, 0), (0, 0), (pb, pt), (0, 0)]
            t.image = np.pad(t.image, padding)
            t.extent[4:] = [z_min, z_max]

        return model.Piece(tiles, piece, copied_from=copied)

    def _load_collection(self):
        pieces = [self.load_piece(p, sn) for p, sn in self.list_pieces().items()]
        if len(pieces) == 0:
            raise IOError(f'No pieces found in {self.path}')
        return model.Cochlea(pieces)

    def list_pieces(self):
        raise NotImplementedError

    def _load_tile(self, stack_name):
        raise NotImplementedError

    def save_path(self):
        raise NotImplementedError


class LIFCochleaReader(CochleaReader):
    '''
    Reads an entire cochlea from a LIF file.

    This expects multiple images with at least one image per piece, possibly
    more if the entire piece did not fit inside the field of view. All images
    should be saved to the same LIF file and contain the piece numbers.
    '''
    def __init__(self, path):
        from readlif.reader import LifFile
        super().__init__(path)
        self.fh = LifFile(path)

    def list_pieces(self):
        p_piece = re.compile(r'^(?!_)piece_(\d+)\w?')
        pieces = {}
        for img in self.fh.get_iter_image():
            try:
                piece = int(p_piece.match(img.name).group(1))
                pieces.setdefault(piece, []).append(img.name)
            except Exception as e:
                pass
        return {p: pieces[p] for p in sorted(pieces)}

    def _load_tile(self, stack_name):
        info, img = util.load_lif(self.path, stack_name)
        name = f'{self.path.stem}_{stack_name}'
        return model.Tile(info, img, name)

    def save_path(self):
        return self.path.parent / self.path.stem


class CZICochleaReader(CochleaReader):
    '''
    Reads an entire cochlea from a folder containing multiple CZI files (one
    image per file).

    This expects multiple images with at least one image per piece, possibly
    more if the entire piece did not fit inside the field of view. All images
    should be saved to the same folder and contain the piece numbers.
    '''
    def list_pieces(self):
        p_piece = re.compile(r'^(?!_).*piece_(\d+)\w?')
        pieces = {}
        for filename in self.path.glob('*piece_*.czi'):
            try:
                piece = int(p_piece.match(filename.stem).group(1))
                pieces.setdefault(piece, []).append(filename.stem)
            except Exception as e:
                pass
        return {p: pieces[p] for p in sorted(pieces)}

    def _load_tile(self, stack_name):
        filename = self.path / f'{stack_name}.czi'
        info, img = util.load_czi(filename)
        name = f'{self.path.stem}_{stack_name}'
        return model.Tile(info, img, name)

    def save_path(self):
        return self.path.parent / self.path.stem


class ProcessedCochleaReader(CochleaReader):

    def list_pieces(self):
        p_piece = re.compile(r'.*piece_(\d+)\w?')
        pieces = {}
        for path in self.path.glob('*piece_*.*'):
            if path.name.endswith('.json'):
                continue
            piece = int(p_piece.match(path.stem).group(1))
            pieces.setdefault(piece, []).append(path.stem)
        return {p: pieces[p] for p in sorted(pieces)}

    def _load_tile(self, stack_name):
        image = np.load(f'{stack_name}.npy')
        info = json.loads(filename.with_suffix('.json').read_text())
        return model.Tile(info, image, filename.stem)

    def save_path(self):
        return self.path


class TileReader(BaseReader):
    '''
    Simple reader that loads a tile collection
    '''
    def __init__(self, path, pattern=None):
        if pattern is None:
            pattern = '(.*)'
        self.path = Path(path)
        self.pattern = re.compile(pattern)

    def _load_collection(self):
        tiles = [self.load_tile(r) for r in self.list_tiles()]
        return model.TileAnalysisCollection(tiles=tiles)

    def state_filename(self, obj):
        return self.save_path() / f'{obj.name}_analysis.json'

    def load_tile(self, tile_name):
        raise NotImplementedError

    def list_tiles(self):
        raise NotImplementedError

    def save_path(self):
        raise NotImplementedError


class LIFTileReader(TileReader):

    def __init__(self, path, pattern='(.*OHC.*)'):
        from readlif.reader import LifFile
        super().__init__(path, pattern)
        self.fh = LifFile(path)

    def load_tile(self, tile_name):
        info, img = util.load_lif(self.path, tile_name)
        tile = model.Tile(info, img, source=tile_name)
        freq = extract_frequency(tile_name)
        return model.TileAnalysis(tile, name=tile_name, frequency=freq)

    def list_tiles(self):
        tile_names = {}
        for img in self.fh.get_iter_image():
            try:
                tile_name = self.pattern.match(img.name).group(1)
                tile_names[img.name] = tile_name
            except Exception as e:
                pass
        return sorted(tile_names, key=extract_frequency)

    def save_path(self):
        return self.path.parent / self.path.stem


class CZITileReader(TileReader):

    def load_tile(self, tile_name):
        filename = self.path / f'{tile_name}.czi'
        info, img = util.load_czi(filename)
        tile = model.Tile(info, img, source=tile_name)
        freq = extract_frequency(tile_name)
        return model.TileAnalysis(tile, name=tile_name, frequency=freq)

    def list_tiles(self):
        tile_names = {}
        for img in self.path.glob('*.czi'):
            tile_names[img.stem] = img.stem
        return sorted(tile_names, key=extract_frequency)

    def save_path(self):
        return self.path
