import numpy as np
import pandas as pd

import os
import json
from sklearn.model_selection import KFold


N_splits = 10

if __name__ == '__main__':
    out_path = '/home/dlachinov/cardiospike/splits'

    input_datasets = '/home/dlachinov/cardiospike/data/cardiospike/train.csv'

    dataset = pd.read_csv(input_datasets)
    datasets = ['cardiospike']

    config = {'n_splits':N_splits,
              'datasets':{}}

    for d in datasets:
        split_path = os.path.join(out_path, d)
        os.makedirs(split_path,exist_ok=True)

        ids = dataset['id'].unique()

        kf = KFold(n_splits=10,random_state=1337,shuffle=True)

        for idx,(train_index, test_index) in enumerate(kf.split(ids)):
            X_train, X_test = ids[train_index], ids[test_index]
            np.savetxt(os.path.join(split_path,'train_'+str(idx)+'.csv'), X_train)
            np.savetxt(os.path.join(split_path,'test_'+str(idx)+'.csv'),X_test, delimiter=",")

        config['datasets'][d] = {'data_path':'data/cardiospike/train.csv',
                                 'metadata_path':None,
                                 'splits_path':os.path.join('splits',d)}

    with open('../config_cardispike.cfg', 'w') as outfile:
        json.dump(config, outfile,indent=4)