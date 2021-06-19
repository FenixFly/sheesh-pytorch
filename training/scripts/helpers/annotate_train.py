import pandas as pd
import numpy as np
import os
import tqdm
from skimage import morphology
import torch


import seaborn as sns
from matplotlib import pyplot as plt


def get_patient_data(dataset, id):
    return dataset.loc[dataset['id']==id,:].sort_values(by='time', axis=0, ascending=True)

input_datasets = '/home/dlachinov/cardiospike/data/cardiospike'

dataset = pd.read_csv(os.path.join(input_datasets,'train.csv'))

dataset['pos'] = 0
dataset['centers'] = 0
dataset['std'] = 0

ids = dataset['id'].unique().tolist()


results = []
lengts = []

elem = np.array([0, 50, -50, 0,])

for id in tqdm.tqdm(ids):
    s = get_patient_data(dataset,id)
    local_pos = np.arange(s.shape[0])
    local_centers = np.zeros(s.shape[0])

    labels, num = morphology.label(s['y'],connectivity=1,return_num=True)

    results.append({'id':id,'num':num})


    for i in range(num):
        pdf = labels==(i+1)
        pdf = pdf / pdf.sum()
        idx = int((local_pos*pdf).sum())
        local_centers[idx] = 1

        lengts.append((labels==(i+1)).sum())

    dataset.loc[s.index,'centers'] = local_centers
    dataset.loc[s.index,'pos'] = np.arange(s.shape[0])

    unfolded = torch.from_numpy(np.pad(s['x'].to_numpy(), pad_width=((2, 1),), mode='reflect')).unfold(dimension=0,
                                                                                                       size=4,
                                                                                                       step=1).float()
    mean = torch.mean(unfolded, dim=1).numpy()
    mean2 = torch.mean(unfolded ** 2, dim=1).numpy()
    std = np.sqrt(mean2 - mean ** 2)

    pre = (unfolded.numpy() - mean.reshape((-1, 1)) - elem.reshape((1, -1))) ** 2
    pre = np.mean(pre, axis=1)
    pre = np.clip(pre, 0, 1000)

    dataset.loc[s.index, 'std'] = pre


stat = pd.DataFrame(results)

sns.histplot(data=lengts)
plt.show()

sns.histplot(data=stat['num'])
plt.show()

dataset.to_csv(os.path.join(input_datasets,'processed.csv'))

print('min size', np.min(lengts))