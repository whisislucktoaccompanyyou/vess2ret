"""Dataset loading utilities for paired A/B image translation."""

import os
import random

import numpy as np
from PIL import Image, ImageOps
from keras.utils import Sequence


class TwoImageIterator(Sequence):
    """Yield paired images from ``directory/A`` and ``directory/B``.

    The iterator returns ``[batch_a, batch_b]`` in channels-last format. A
    binary image is scaled to 0/1; a non-binary image is scaled to -1/1 for a
    tanh generator output.
    """

    def __init__(
        self,
        directory,
        a_dir_name="A",
        b_dir_name="B",
        load_to_memory=False,
        is_a_binary=False,
        is_b_binary=False,
        is_a_grayscale=False,
        is_b_grayscale=False,
        target_size=(256, 256),
        rotation_range=0.0,
        height_shift_range=0.0,
        width_shift_range=0.0,
        zoom_range=0.0,
        fill_mode="constant",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        dim_ordering="default",
        N=-1,
        batch_size=32,
        shuffle=True,
        seed=None,
    ):
        if dim_ordering in ("default", "tf"):
            dim_ordering = "tf"
        elif dim_ordering == "th":
            raise ValueError(
                "channels-first ('th') is no longer supported; use the modern "
                "channels-last format ('tf')."
            )
        else:
            raise ValueError('dim_ordering must be "default" or "tf"')

        self.directory = directory
        self.a_dir = os.path.join(directory, a_dir_name)
        self.b_dir = os.path.join(directory, b_dir_name)
        if not os.path.isdir(self.a_dir) or not os.path.isdir(self.b_dir):
            raise FileNotFoundError(
                f"Paired dataset folders were not found: {self.a_dir} and {self.b_dir}"
            )

        a_files = {name for name in os.listdir(self.a_dir) if os.path.isfile(os.path.join(self.a_dir, name))}
        b_files = {name for name in os.listdir(self.b_dir) if os.path.isfile(os.path.join(self.b_dir, name))}
        self.filenames = sorted(a_files.intersection(b_files))
        if N > 0:
            rng = random.Random(seed)
            rng.shuffle(self.filenames)
            self.filenames = self.filenames[:N]
        self.N = len(self.filenames)
        if self.N == 0:
            raise ValueError(
                "Did not find any paired images. Check that A and B contain files "
                f"with the same names: {self.a_dir}, {self.b_dir}"
            )

        self.dim_ordering = dim_ordering
        self.target_size = tuple(target_size)
        self.is_a_binary = is_a_binary
        self.is_b_binary = is_b_binary
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale
        self.load_to_memory = load_to_memory
        self.rotation_range = float(rotation_range)
        self.height_shift_range = float(height_shift_range)
        self.width_shift_range = float(width_shift_range)
        self.zoom_range = self._normalize_zoom_range(zoom_range)
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._cursor = 0
        self.indexes = np.arange(self.N)
        self.on_epoch_end()

        self.image_shape_a = self._get_image_shape(is_a_grayscale)
        self.image_shape_b = self._get_image_shape(is_b_grayscale)
        self.a = None
        self.b = None
        if self.load_to_memory:
            self._load_imgs_to_memory()

    @staticmethod
    def _normalize_zoom_range(zoom_range):
        if np.isscalar(zoom_range):
            zoom = float(zoom_range)
            return (1.0 - zoom, 1.0 + zoom)
        if len(zoom_range) != 2:
            raise ValueError("zoom_range must be a scalar or a pair")
        return (float(zoom_range[0]), float(zoom_range[1]))

    def _get_image_shape(self, is_grayscale):
        return self.target_size + (1 if is_grayscale else 3,)

    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(self.N)
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __iter__(self):
        while True:
            yield self.__next__()

    def __next__(self):
        if self._cursor >= self.N:
            self._cursor = 0
            self.on_epoch_end()
        start = self._cursor
        stop = min(start + self.batch_size, self.N)
        self._cursor = stop
        return self._make_batch(self.indexes[start:stop])

    def __getitem__(self, batch_index):
        start = int(batch_index) * self.batch_size
        stop = min(start + self.batch_size, self.N)
        if start >= self.N:
            raise IndexError(batch_index)
        return self._make_batch(self.indexes[start:stop])

    def _load_imgs_to_memory(self):
        self.a = np.zeros((self.N,) + self.image_shape_a, dtype=np.float32)
        self.b = np.zeros((self.N,) + self.image_shape_b, dtype=np.float32)
        for idx in range(self.N):
            self.a[idx], self.b[idx] = self._load_img_pair(idx, False)

    def _load_img_pair(self, idx, load_from_memory):
        if load_from_memory:
            return self.a[idx].copy(), self.b[idx].copy()
        filename = self.filenames[idx]
        a = self._read_image(os.path.join(self.a_dir, filename), self.is_a_grayscale)
        b = self._read_image(os.path.join(self.b_dir, filename), self.is_b_grayscale)
        return a, b

    def _read_image(self, path, grayscale):
        mode = "L" if grayscale else "RGB"
        with Image.open(path) as image:
            image = image.convert(mode)
            image = image.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
        if grayscale:
            array = array[..., np.newaxis]
        return array

    def _random_transform(self, a, b):
        """Apply the same lightweight augmentation to both paired images."""
        if self.rotation_range:
            angle = float(self.rng.uniform(-self.rotation_range, self.rotation_range))
            a = self._rotate_array(a, angle)
            b = self._rotate_array(b, angle)

        if self.horizontal_flip and self.rng.random() < 0.5:
            a = np.flip(a, axis=1).copy()
            b = np.flip(b, axis=1).copy()
        if self.vertical_flip and self.rng.random() < 0.5:
            a = np.flip(a, axis=0).copy()
            b = np.flip(b, axis=0).copy()
        return a, b

    def _rotate_array(self, array, angle):
        channels = array.shape[-1]
        mode = "L" if channels == 1 else "RGB"
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode=mode)
        fill = int(np.clip(self.cval, 0, 255))
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)
        result = np.asarray(image, dtype=np.float32)
        if channels == 1:
            result = result[..., np.newaxis]
        return result

    @staticmethod
    def _binarize(batch):
        return (batch / 255.0 >= 0.5).astype(np.float32)

    @staticmethod
    def _normalize_for_tanh(batch):
        return (batch - 127.5) / 127.5

    def _make_batch(self, indexes):
        batch_a = np.zeros((len(indexes),) + self.image_shape_a, dtype=np.float32)
        batch_b = np.zeros((len(indexes),) + self.image_shape_b, dtype=np.float32)
        for batch_index, image_index in enumerate(indexes):
            a, b = self._load_img_pair(int(image_index), self.load_to_memory)
            batch_a[batch_index], batch_b[batch_index] = self._random_transform(a, b)

        if self.is_a_binary:
            batch_a = self._binarize(batch_a)
        else:
            batch_a = self._normalize_for_tanh(batch_a)
        if self.is_b_binary:
            batch_b = self._binarize(batch_b)
        else:
            batch_b = self._normalize_for_tanh(batch_b)
        return [batch_a, batch_b]
