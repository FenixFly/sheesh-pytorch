import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import tqdm

import os


def get_patient_data(dataset, id):
    return dataset.loc[dataset['id'] == id, :].sort_values(by='time', axis=0, ascending=True)


def plot(data, mask=None, color=None):
    x = np.arange(0, data.shape[0])
    plt.plot(x, data,color=(0,0,1,1))
    if mask is not None:
        plt.fill_between(x, 0, 1400, where=mask, color=color)


if __name__ == '__main__':
    plot_save_path = '/home/dlachinov/cardiospike/data/cardiospike/plots'

    dataset = pd.read_csv('/home/dlachinov/cardiospike/scripts/002c/cv_results_c1.csv')#'/home/dlachinov/cardiospike/data/cardiospike/train.csv')

    ids = dataset['id'].unique().tolist()

    for id in tqdm.tqdm(ids):
        signal = get_patient_data(dataset,id)
        plt.figure(figsize=(100, 3), dpi=100)
        plot(signal['x'],mask=signal['y']>0.5,color=(1, 0, 0, 0.5))

        if 'y_pred' in signal:
            plot(signal['x'], mask=signal['y_pred']>0.5,color=(0, 1, 0, 0.5))

        plt.ylim(0,1200)
        plt.savefig(fname=os.path.join(plot_save_path,'{0:03d}'.format(id)+'.svg'))
        plt.close()
    print('fin')