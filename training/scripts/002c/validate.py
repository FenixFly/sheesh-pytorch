import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt
import math
import re
import pandas as pd
import pickle
import os
import scipy.stats as stats
from torch.utils.data import DataLoader, ConcatDataset
import copy
import seaborn as sns
import json
from PIL import Image
import cv2

from common import loader_helper
from common import mytransforms
from common import metrics, blocks
from common import dataloader
from common import pl_model_wrapper


parser = argparse.ArgumentParser(description="PyTorch CardioSpike Val")
parser.add_argument("--root_dir", default="", type=str, help="project root dir")
parser.add_argument("--cv_configuration", default="", type=str, help="path to the config")
parser.add_argument("--split", nargs='+', default=None, type=int, help="splits to train")
parser.add_argument("--name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default=None, type=str, help="path to models folder")

import model
n_outputs = 1

arch = model.CustomLSTM(input_feature_size=1, hidden_size=64, n_outputs=n_outputs)
params = {'model': arch, 'losses':None, 'metrics':None, 'metametrics':None, 'optim':None}

def get_checkpoint_path(path, start_epoch=0, n_from_best=0):
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    rx = re.compile('epoch=(\d+)-Validation_F1=(\d+\.\d+)\.ckpt')
    results = []

    for f in filenames:
        g = rx.search(f)
        r = {'epoch':int(g.group(1)),
             'val':float(g.group(2))}
        results.append(r)

    results = pd.DataFrame(results)

    results = results[results['epoch']>start_epoch].sort_values(by='val',ascending=False)

    return 'epoch={:d}-Validation_F1={:.4f}.ckpt'.format(int(results.iloc[n_from_best,:]['epoch']),
                                                       results.iloc[n_from_best,:]['val'])



if __name__ == '__main__':
    import model
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)

    data_transform_val = mytransforms.Compose([
        mytransforms.Normalization(transform_keys=['signal'], mean=617.7666, std=287.5535),
        #mytransforms.PadToRate(transform_keys=['signal', 'label', 'time', 'x'], rate=[1, 64]),
        # mytransforms.ZScoreNormalization(transform_keys=['signal'], axis=(1,)),
        mytransforms.ToTensorDict(transform_keys=['signal', 'label', 'time'])
    ])

    metrics_val = {
        'F1': metrics.F1micro(output_key='prediction', target_key='label', slice=0),
        'F1global': metrics.F1global(output_key='prediction', target_key='label', slice=0)
    }

    with open(opt.cv_configuration) as json_file:
        cv_config = json.load(json_file)


    split_list = [f for f in os.listdir(opt.models_path) if os.path.isdir(os.path.join(opt.models_path,f))]
    split_idx = [int(f.split('_')[1]) for f in split_list]
    split_idx.sort()

    results = []

    for idx, f in enumerate(split_idx):
        split_opt = copy.deepcopy(opt)

        f = 'split_' + str(idx) + '.p'

        split_opt.name = opt.name + '_' + str(idx)
        model_path = os.path.join(split_opt.models_path, split_opt.name)
        split_opt.models_path = model_path

        datasets_config = cv_config['datasets']


        path_weights = get_checkpoint_path(split_opt.models_path,start_epoch=20,n_from_best=2)

        test_datasets = []
        for k, v in datasets_config.items():
            test_items = np.loadtxt(os.path.join(opt.root_dir,v['splits_path'],'test_'+str(idx)+'.csv'),dtype=str).tolist()
            test_dataset = dataloader.CardioSpikeDataset(path=os.path.join(opt.root_dir, v['data_path']),
                                                     patients=test_items,
                                                     multiplier=1,
                                                     transforms=data_transform_val,
                                                     validation=True)
            test_datasets.append(test_dataset)

        val_data = ConcatDataset(test_datasets)

        evaluation_data_loader = DataLoader(dataset=val_data, num_workers=1, batch_size=1, shuffle=False,
                                            drop_last=False)


        model_trainer = pl_model_wrapper.Model.load_from_checkpoint(os.path.join(split_opt.models_path,path_weights),**params)

        model_trainer.eval()
        torch.no_grad()
        for batch in evaluation_data_loader:

            output = model_trainer(batch)


            output = output['prediction'][0].cpu().detach().numpy()
            time = batch['time'][0].cpu().detach().numpy()
            x = batch['x'][0].cpu().detach().numpy()
            y = batch['label'][0].cpu().detach().numpy()

            scan_result = pd.DataFrame(data={'id':batch['id'][0].item(),
                               'time':time.astype(np.int)[0],'x':x[0],'y':y[0],'y_pred':output[0],'split':idx})
            results.append(scan_result)

        print(str(idx), 'finished')




    df_results = pd.concat(results,axis=0)
    df_results = df_results.sort_values(by=['id','time'],ascending=True)
    df_results.to_csv('cv_results_c3.csv')


