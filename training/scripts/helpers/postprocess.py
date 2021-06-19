import numpy as np
import pandas as pd
from skimage import morphology
import torch


elem = np.array([0, 50, -50, 0,])


def run_filter(sample):
    unfolded = torch.from_numpy(np.pad(sample, pad_width=((2, 1),), mode='reflect')).unfold(dimension=0,
                                                                                                       size=4,
                                                                                                       step=1).float()
    mean = torch.mean(unfolded, dim=1).numpy()
    pre = (unfolded.numpy() - mean.reshape((-1, 1)) - elem.reshape((1, -1))) ** 2
    pre = np.mean(pre, axis=1)
    pre = np.clip(pre, 0, 1000)
    return pre



def postprocess(sample):

    labels, num = morphology.label(sample, connectivity=1, return_num=True)

    filtered = run_filter(sample)

    small_samples = 0
    erased = 0
    dilated = 0

    for i in range(num):
        pdf = labels==(i+1)

        len = np.sum(pdf)

        if len < 6:

            if (filtered[pdf] < 200).sum() ==0 and len<4:
                sample[pdf] = 0
                erased +=1
            else:
                pdf = morphology.dilation(pdf,selem=[1,1,1])
                sample[pdf] = 1
                dilated += 1
            small_samples +=1

    print('small intervals', small_samples, 'erased', erased, 'dilated', dilated)

    return sample