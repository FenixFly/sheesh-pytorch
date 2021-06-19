import numpy as np


class DuplicateError(Exception):
    def __init__(self, key, message):
        self.key = key
        self.message = message

def get_key(parent_key, key):
    if parent_key:
        key = parent_key + '.' + key
    return key

def pad_to_shape(image, shape, mode='minimum'):
    diff = np.array(shape) - np.array(image.shape)

    diff[np.isnan(diff)] = 0
    diff = diff.astype(np.int32)

    pad_left = diff // 2
    pad_right = diff - pad_left
    padded = np.pad(image, pad_width=tuple(zip(pad_left, pad_right)), mode=mode)
    return padded, (pad_left, pad_right)

def revert_pad(image, padding, dim:list=None):
    pad_left, pad_right = padding

    if dim is None:
        dim = range(len(image.shape))

    start = pad_left
    end = [s - r for s, r in zip(image.shape,pad_right)]

    index = tuple([slice(s, e) if idx in dim else slice(None) for idx,(s, e) in enumerate(zip(start, end))])

    return image[index]

def closest_to_k(n,k=8):
    if n % k == 0:
        return n
    else:
        return ((n // k) + 1)*k