import logging

log = logging.getLogger(__name__)

import json
from pathlib import Path
import re

import numpy as np

from . import model
from . import util


class BaseReader:

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

    def __init__(self, path):
        self.path = Path(path)

    def save_figure(self, fig, suffix):
        filename = self.save_path() / f'{self.get_name()}_{suffix}.pdf'
        filename.parent.mkdir(exist_ok=True)
        fig.savefig(filename)

    def state_filename(self, piece):
        return self.save_path() / f'{self.path.stem}_piece_{piece.piece}_analysis.json'


class LIFCochleaReader(CochleaReader):

    def __init__(self, path):
        from readlif.reader import LifFile
        super().__init__(path)
        self.fh = LifFile(path)

    def list_pieces(self):
        p_piece = re.compile('^(?!_)piece_(\d+)\w?')
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

    def save_path(self):
        return self.path.parent / self.path.stem


class ProcessedCochleaReader(CochleaReader):

    def list_pieces(self):
        p_piece = re.compile('.*piece_(\d+)\w?')
        pieces = []
        for path in self.path.glob('*piece_*.*'):
            if path.name.endswith('.json'):
                continue
            piece = int(p_piece.match(path.stem).group(1))
            pieces.append(piece)
        return sorted(set(pieces))

    def _load_tile(self, filename):
        image = np.load(filename)
        info = json.loads(filename.with_suffix('.json').read_text())
        return model.Tile(info, image, filename.stem)

    def load_piece(self, piece):
        tile_filenames = sorted(self.path.glob(f"*piece_{piece}*.npy"))
        log.info('Found tiles: %r', [t.stem for t in tile_filenames])
        tiles = [self._load_tile(f) for f in tile_filenames]

        copy = re.compile(fr'^.*piece_{piece}\w?_copied_([\w-]+)')
        copied = set()
        for tf in tile_filenames:
            if (m := copy.match(tf.stem)) is not None:
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
        pieces = [self.load_piece(p) for p in self.list_pieces()]
        if len(pieces) == 0:
            raise IOError(f'No pieces found in {self.path}')
        return model.Cochlea(pieces)

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


class LIFTileReader(TileReader):

    def __init__(self, path, pattern='(.*OHC.*)'):
        from readlif.reader import LifFile
        super().__init__(path, pattern)
        self.fh = LifFile(path)

    def _load_collection(self):
        tiles = [self.load_tile(r) for r in self.list_tiles()]
        return model.TileAnalysisCollection(tiles=tiles)

    def load_tile(self, tile_name):
        info, img = util.load_lif(self.path, tile_name)
        tile = model.Tile(info, img, source=tile_name)
        return model.TileAnalysis(tile, name=tile_name)

    def list_tiles(self):
        tile_names = {}
        for img in self.fh.get_iter_image():
            try:
                tile_name = self.pattern.match(img.name).group(1)
                tile_names[img.name] = tile_name
            except Exception as e:
                pass
        return tile_names

    def save_path(self):
        return self.path.parent / self.path.stem

    def state_filename(self, obj):
        return self.save_path() / f'{obj.name}_analysis.json'
