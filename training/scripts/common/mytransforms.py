import numpy as np
import torch
import random
import math
import cv2

from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import loader_helper

class Transform:
    def __init__(self, transform_keys:list):
        self.transform_keys = transform_keys

    def __call__(self, data:dict):
        raise NotImplementedError()

class Compose:
    def __init__(self, transforms:list):
        self.transforms = transforms

    def __call__(self, data:dict):
        results = data
        for t in self.transforms:
            results = t(data)
        return results

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        results = args
        for t in self.transforms:
            results = t(*results)
        return results


class RandomCrop(Transform):
    def __init__(self, transform_keys:list, size):
        super(RandomCrop, self).__init__(transform_keys)
        self.size = np.array(size)


    def get_index(self,array,start_random):
        rel_size = np.minimum(self.size / np.array(array.shape), 1)
        start = (1. - rel_size) * start_random

        save_size_index = np.isnan(self.size)


        start = np.floor(start * array.shape)
        end = np.minimum(start + self.size, array.shape)

        start[save_size_index] = 0
        end[save_size_index] = np.array(array.shape)[save_size_index]

        index = tuple([slice(s, e) for s, e in zip(start.astype(np.int32), end.astype(np.int32))])

        return index

    def __call__(self, data:dict):

        shape = data[self.transform_keys[0]].shape

        #make cropping relative to the size

        start_random = np.random.random(size=(len(shape),))
        #.random.randint(0, np.maximum(max - crop, 1)) for max, crop in zip(shape,self.size)]


        for key in self.transform_keys:
            #assert len(data[key].shape) == len(shape)
            #assert np.all(data[key].shape == shape)



            if isinstance(data[key],dict):
                for subkey in data[key]:
                    index = self.get_index(data[key][subkey],start_random)
                    data[key][subkey] = data[key][subkey][index]
            else:
                index = self.get_index(data[key],start_random)
                data[key] = data[key][index].copy()

        return data


class NotSoRandomCrop(RandomCrop):
    def __init__(self, transform_keys:list, guide_key, p, size):
        super(NotSoRandomCrop, self).__init__(transform_keys,size)
        self.size = np.array(size)
        self.guide_key = guide_key
        self.p = p


    def get_index_guide(self,array,guide,start_random):

        sample_from = np.where(guide>0.5)[1]
        sample_start = np.floor(start_random*sample_from.shape[0])


        #start = (1. - rel_size) * start_random

        #save_size_index = np.isnan(self.size)


        start = np.minimum(sample_from[int(sample_start[1])], array.shape[1]-self.size[1]) #np.floor(start * array.shape)
        end = np.minimum(start + self.size, array.shape)

        index = (slice(0,array.shape[0]),slice(int(start),int(end[1])))#tuple([slice(s, e) for s, e in zip(start.astype(np.int32), end.astype(np.int32))])

        return index

    def __call__(self, data:dict):

        shape = data[self.transform_keys[0]].shape

        #make cropping relative to the size

        start_random = np.random.random(size=(len(shape),))
        #.random.randint(0, np.maximum(max - crop, 1)) for max, crop in zip(shape,self.size)]
        guide = data[self.guide_key]

        for key in self.transform_keys:
            #assert len(data[key].shape) == len(shape)
            #assert np.all(data[key].shape == shape)



            if isinstance(data[key],dict):
                for subkey in data[key]:
                    index = self.get_index_guide(data[key][subkey], guide,start_random)
                    data[key][subkey] = data[key][subkey][index]
            else:
                index = self.get_index_guide(data[key], guide,start_random)
                data[key] = data[key][index].copy()

        return data


import collections

class PadToShape(Transform):
    def __init__(self, transform_keys: list, shape, mode='minimum'):
        super(PadToShape, self).__init__(transform_keys)
        self.shape = shape
        self.mode = mode

    def __call__(self, data: dict):

        for key in self.transform_keys:
            #assert len(data[key].shape) == len(data_shape)
            #assert np.all(data[key].shape == data_shape)


            if isinstance(data[key],(dict,collections.OrderedDict)):
                if data[key]:
                    data_shape = data[key][list(data[key].keys())[0]].shape
                    for subkey in data[key]:
                        padded, (pad_left, pad_right) = loader_helper.pad_to_shape(data[key][subkey], self.shape, self.mode)
                        data['padding' + key] = (pad_left, pad_right)
                        data[key][subkey] = padded
                    data['shape_old' + key] = data_shape
                    data['shape_new' + key] = self.shape
            else:

                data_shape = data[key].shape
                padded,(pad_left, pad_right) = loader_helper.pad_to_shape(data[key],self.shape)

                data['shape_old'+key] = data_shape
                data['shape_new'+key] = self.shape
                data['padding'+key] = (pad_left, pad_right)
                data[key] = padded

        return data

class PadToRate(Transform):
    def __init__(self, transform_keys: list, rate:list, dim:list=None):
        super(PadToRate, self).__init__(transform_keys)
        self.rate = np.array(rate)
        self.dim = dim

    def pad_to_rate(self,array,key):
        data_shape = array.shape
        if self.dim is None:
            self.dim = list(range(len(data_shape)))

        new_shape = [loader_helper.closest_to_k(i, r) if idx in self.dim else i for idx, (i, r) in
                     enumerate(zip(data_shape, self.rate))]
        new_shape = tuple(new_shape)

        padded, (pad_left, pad_right) = loader_helper.pad_to_shape(array, new_shape)


        desc = {}
        desc['shape_old' + key] = data_shape
        desc['shape_new' + key] = new_shape
        desc['padding' + key] = (pad_left, pad_right)
        return padded, desc

    def __call__(self, data: dict):
        #data_shape = data[self.transform_keys[0]].shape

        for key in self.transform_keys:
            #assert len(data[key].shape) == len(data_shape)
            #assert np.all(data[key].shape == data_shape)

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    array, desc = self.pad_to_rate(data[key][subkey], key)
                    data[key][subkey] = array
                    data.update(desc)
            else:
                array, desc = self.pad_to_rate(data[key],key)
                data[key] = array
                data.update(desc)

        return data

class RandomMirror(Transform):
    def __init__(self, transform_keys:list, dimensions:list):
        super(RandomMirror, self).__init__(transform_keys)
        self.dimensions = dimensions

    def flip(self, image, p):

        index = [slice(0, size) for size in image.shape]

        for i in self.dimensions:
            if p[i] < 0.5:
                index[i] = slice(-1, -image.shape[i] - 1, -1)

        index = tuple(index)

        return image[index].copy()

    def __call__(self, data:dict):

        #print('TEST,crop',shape,self.size,[(max - crop or 1) for max, crop in zip(shape,self.size)])
        dim = len(data[self.transform_keys[0]].shape)

        p = np.random.random(dim)

        for key in self.transform_keys:

            if isinstance(data[key],dict):
                for subkey in data[key]:
                    data[key][subkey] = self.flip(data[key][subkey],p)

            else:
                data[key] = self.flip(data[key],p)

        return data

class ZScoreNormalization(Transform):
    def __init__(self, transform_keys:list, axis):
        super(ZScoreNormalization, self).__init__(transform_keys)
        self.axis = axis

    def __call__(self, data:dict):
        #mean = 9310.
        #std = 8759
        for key in self.transform_keys:
            mean = data[key].mean(axis=self.axis,keepdims=True)
            std = data[key].std(axis=self.axis,keepdims=True)

            data[key] = (data[key] - mean + 1)/(std+1)

        return data

class Normalization(Transform):
    def __init__(self, transform_keys:list, mean, std):
        super(Normalization, self).__init__(transform_keys)
        self.mean = mean
        self.std = std

    def __call__(self, data:dict):
        for key in self.transform_keys:
            data[key] = (data[key] - self.mean)/(self.std)

        return data

class IntensityShift(Transform):
    def __init__(self, transform_keys:list, min:float = -0.6, max:float = 0.6):
        super(IntensityShift, self).__init__(transform_keys)
        self.min = min
        self.max = max

    def __call__(self, data:dict):

        for key in self.transform_keys:

            data[key] = data[key] + random.uniform(self.min, self.max)

        return data

class ContrastAugmentation(Transform):
    def __init__(self, transform_keys: list, min: float = 0.6, max: float = 1.4):
        super(ContrastAugmentation, self).__init__(transform_keys)
        self.min = min
        self.max = max

    def __call__(self, data: dict):
        for key in self.transform_keys:
            data[key] = data[key] * random.uniform(self.min, self.max)

class ToTensorDict(Transform):
    def __init__(self, transform_keys: list):
        super(ToTensorDict, self).__init__(transform_keys)

    def __call__(self, data: dict):
        for key in self.transform_keys:


            if isinstance(data[key],dict):
                for subkey in data[key]:
                    data[key][subkey] = torch.from_numpy(data[key][subkey]).float()

            else:
                data[key] = torch.from_numpy(data[key]).float()

        return data


class NoiseAugmentation(Transform):
    def __init__(self, transform_keys, sigma = 2.0, prob = 0.2):
        super(NoiseAugmentation, self).__init__(transform_keys)
        self.sigma = sigma
        self.prob = prob

    def __call__(self, data: dict):

        for key in self.transform_keys:
            if random.random() < self.prob:
                data[key] = data[key] + np.random.normal(0, self.sigma, size=data[key].shape)

        return data


class ToTensor:
    def __call__(self, *args):

        results = []

        for a in args:
            img = torch.from_numpy(a).float()
            results.append(img)

        return tuple(results)