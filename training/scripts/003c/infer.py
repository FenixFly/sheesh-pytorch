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
parser.add_argument("--data_path", default="", type=str, help="path to csv")
parser.add_argument("--split", nargs='+', default=None, type=int, help="splits to infer")
parser.add_argument("--name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default=None, type=str, help="path to models folder")

import model

n_outputs = 1

arch = model.CustomTransformer(input_feature_size=1, hidden_size=64, n_outputs=n_outputs, n_head=4, n_encoders=4)
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
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)

    data_transform_val = mytransforms.Compose([
        mytransforms.Normalization(transform_keys=['signal'], mean=617.7666, std=287.5535),
        # mytransforms.ZScoreNormalization(transform_keys=['signal'], axis=(1,)),
        mytransforms.ToTensorDict(transform_keys=['signal'])
    ])

    opt.models_path = os.path.join(opt.models_path, opt.name)
    split_list = [f for f in os.listdir(opt.models_path) if os.path.isdir(os.path.join(opt.models_path,f))]
    split_idx = [int(f.split('_')[1]) for f in split_list]
    split_idx.sort()

    results = []


    data = pd.read_csv(opt.data_path)
    ids = data['id'].unique()

    for idx, f in enumerate(split_idx):
        split_opt = copy.deepcopy(opt)

        f = 'split_' + str(idx) + '.p'

        split_opt.name = opt.name + '_' + str(idx)
        model_path = os.path.join(split_opt.models_path, split_opt.name)
        split_opt.models_path = model_path


        path_weights = get_checkpoint_path(split_opt.models_path,start_epoch=20,n_from_best=0)


        model_trainer = pl_model_wrapper.Model.load_from_checkpoint(os.path.join(split_opt.models_path,path_weights),**params)

        model_trainer.eval()
        torch.no_grad()

        fold_output_name = 'result_'+opt.name+'_split_'+str(idx)
        data[fold_output_name] = 0

        for id in ids:

            signal = data[data['id']==id].sort_values(by='time', axis=0, ascending=True)

            batch = {'signal': signal['x'].to_numpy()[None]}
            batch = data_transform_val(batch)
            batch['signal'] = batch['signal'][None]

            output = model_trainer(batch)

            output = output['prediction'][0,0].cpu().detach().numpy()
            data.loc[signal.index,fold_output_name] = output

        print(str(idx), 'finished')




    data.to_csv('prediction_'+opt.name+'.csv')


