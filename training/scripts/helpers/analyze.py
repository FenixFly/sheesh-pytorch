import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import tqdm
from scipy import signal
import torch
import torch.nn.functional as F

import os

def get_patient_data(dataset, id):
    return dataset.loc[dataset['id']==id,:].sort_values(by='time', axis=0, ascending=True)

def plot(data, mask=None):

    x=np.arange(0,data.shape[0])
    plt.figure(figsize=(25,3),dpi=300)
    plt.plot(x,data)
    if mask is not None:
        plt.fill_between(x,data.min(),data.max(),where=mask,color=(1,0,0,0.5))


if __name__ == '__main__':
    plot_save_path = '/home/dlachinov/cardiospike/data/cardiospike/plots'

    dataset = pd.read_csv('/home/dlachinov/cardiospike/data/cardiospike/train.csv')

    ids = dataset['id'].unique().tolist()

    elem = np.array([ 0, 50,-50, 0, ])

    for id in tqdm.tqdm(ids):
        s = get_patient_data(dataset,id)
        #cc = signal.correlate(s['x'], elem, method='direct')
        #cc = np.clip(cc, -200, 200)
        plot(s['x'],mask=s['y'])

        unfolded = torch.from_numpy(np.pad(s['x'].to_numpy(), pad_width=((2, 1),), mode='reflect')).unfold(dimension=0,
                                                                                                           size=4,
                                                                                                           step=1).float()
        mean = torch.mean(unfolded, dim=1).numpy()
        mean2 = torch.mean(unfolded ** 2, dim=1).numpy()
        std = np.sqrt(mean2 - mean**2)

        pre = (unfolded.numpy() - mean.reshape((-1,1)) - elem.reshape((1,-1)))**2
        pre = np.mean(pre,axis=1)
        pre = np.clip(pre, -1000, 1000)
        plt.plot(np.arange(s['x'].shape[0]), pre)


        #plt.plot(np.arange(cc.shape[0]),cc)
        #plt.ylim(0,1200)
        plt.savefig(fname=os.path.join(plot_save_path,'{0:03d}'.format(id)+'.svg'))
        plt.close()

    sns.histplot(dataset,x='x',hue='y',binrange=(0,1400))
    plt.show()

    ax = sns.boxplot(data=dataset,y='x',x='y')
    ax.set_ylim(0,1400)
    plt.show()

    dataset['x_diff'] = np.clip(dataset['x'].diff(),-200,200)

    s = get_patient_data(dataset, 1)

    unfolded = torch.from_numpy(np.pad(s['x'].to_numpy(),pad_width=((2,2),),mode='reflect')).unfold(dimension=0,size=5,step=1).float()
    mean = torch.mean(unfolded,dim=1).numpy()
    mean2 = torch.mean(unfolded**2,dim=1).numpy()
    std = np.sqrt(mean2 - mean)

    normalized = (s['x'] - mean)/std
    plt.plot(np.arange(s['x'].shape[0]),normalized)
    plt.show()

    sns.histplot(dataset, x='x_diff', hue='y')
    plt.show()

    print(dataset['x'].mean(),dataset['x'].std())

    print(dataset['x_diff'].mean(),dataset['x_diff'].std())



    print('fin')