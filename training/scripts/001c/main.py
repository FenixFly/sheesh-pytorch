import sys

print(sys.path)

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import copy
import json
import pandas as pd
import numpy as np
import shutil
import random

from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import model
from common import dataloader
from common import loader_helper
from common import pl_model_wrapper
from common import blocks
from common import loss
from common import metrics
from common import weight_init
from common import mytransforms


parser = argparse.ArgumentParser(description="PyTorch CardioSpike")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--virtualBatchSize", type=int, default=1, help="virual training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 0")
parser.add_argument("--root_dir", default="", type=str, help="project root dir")
parser.add_argument("--cv_configuration", default="", type=str, help="path to the config")
parser.add_argument("--split", nargs='+', default=None, type=int, help="splits to train")
parser.add_argument("--name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default=None, type=str, help="path to models folder")
parser.add_argument("--gpus", default=1, type=int, help="number of gpus")



def worker_init_fn(worker_id):
    seed = torch.initial_seed() + worker_id
    np.random.seed([int(seed%0x80000000), int(seed//0x80000000)])
    torch.manual_seed(seed)
    random.seed(seed)
    print('worker id {} seed {}'.format(worker_id, seed))


def parse():
    return parser.parse_args()


def main(opt=None, cv_config=None, split=None):
    print(opt)
    print(torch.__version__)
    print(torch.backends.cudnn.version())
    opt.seed = 1337
    opt.multiplier = 10
    pl.seed_everything(1234)
    n_outputs = 1

    print("===> Building model")
    layers = [2, 2, 4]
    number_of_channels = [int(16 * 1 * 2 ** i) for i in range(0, len(layers))]  # [16,16,16,32,32,64,64,64,64,16]

    arch = model.CustomNet(depth=len(layers), encoder_layers=layers, number_of_channels=number_of_channels,
                           number_of_outputs=n_outputs, block=blocks.Residual_bottleneck)
    arch.apply(weight_init.weight_init)

    data_transform = mytransforms.Compose([
        #mytransforms.ZScoreNormalization(transform_keys=['signal'],axis=(1,)),
        mytransforms.Normalization(transform_keys=['signal'],mean=617.7666,std=287.5535),
        mytransforms.RandomCrop(transform_keys=['signal','label','time','x'], size=(np.nan,32)),
        mytransforms.PadToShape(transform_keys=['signal','label','time','x'], shape=(np.nan,32),mode='reflect'),
        #mytransforms.NoiseAugmentation(transform_keys=['signal'],sigma=1.0,prob=0.2),
        #mytransforms.ContrastAugmentation(transform_keys=['signal'],min=0.8, max=1.2),
        #mytransforms.IntensityShift(transform_keys=['signal'],min=-0.2, max=0.2),
        mytransforms.ToTensorDict(transform_keys=['signal','label'])
    ])

    data_transform_val = mytransforms.Compose([
        mytransforms.Normalization(transform_keys=['signal'],mean=617.7666,std=287.5535),
        mytransforms.PadToRate(transform_keys=['signal','label'],rate=[1,32]),
        #mytransforms.ZScoreNormalization(transform_keys=['signal'], axis=(1,)),
        mytransforms.ToTensorDict(transform_keys=['signal','label'])
    ])

    criterion = loss.Mix(losses={
        'BCE': loss.BCE_Loss(output_key='prediction', target_key='label'),
        'Dice': loss.Dice_loss_joint(output_key='prediction', target_key='label'),
    })

    #metrics_train = {
    #    'ROC AUC ' + str(i) : metrics.ROCAUC(output_key='prediction',target_key='label',slice=i)
    #                  for i in range(30)
    #

    metrics_val = {
        'F1' : metrics.F1micro(output_key='prediction',target_key='label',slice=0),
        'F1global' : metrics.F1global(output_key='prediction',target_key='label',slice=0)
    }

    meta_metric_val = None


    train_datasets = []
    test_datasets = []
    datasets_config = cv_config['datasets']

    for k,v in datasets_config.items():
        train_items = np.loadtxt(os.path.join(opt.root_dir,v['splits_path'],'train_'+str(split)+'.csv'),dtype=str).tolist()
        test_items = np.loadtxt(os.path.join(opt.root_dir,v['splits_path'],'test_'+str(split)+'.csv'),dtype=str).tolist()
        train_dataset = dataloader.CardioSpikeDataset(path=os.path.join(opt.root_dir,v['data_path']),
                                            patients=train_items,
                                            multiplier=opt.multiplier,
                                            transforms=data_transform,
                                            validation=False)
        test_dataset = dataloader.CardioSpikeDataset(path=os.path.join(opt.root_dir,v['data_path']),
                                            patients=test_items,
                                            multiplier=1,
                                            transforms=data_transform_val,
                                            validation=True)

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

        print('Dataset', k, 'Training', len(train_dataset), 'Testing', len(test_dataset))

    train_data = ConcatDataset(train_datasets)
    val_data = ConcatDataset(test_datasets)

    print('Total', 'Training', len(train_data), 'Testing', len(val_data))


    training_data_loader = DataLoader(dataset=train_data, num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=True, drop_last=True)

    evaluation_data_loader = DataLoader(dataset=val_data, num_workers=opt.threads//4, batch_size=1, shuffle=False, drop_last=False)

    logger = TensorBoardLogger(
        save_dir=os.path.join(opt.models_path,opt.name),
        version=None,
        name='logs'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(opt.models_path,opt.name),
        save_top_k=-1,
        filename='{epoch}-{Validation_F1:.4f}'
        #save_top_k=5,
        #verbose=True,
        #monitor='Validation/Dice 0',
        #mode='max',
        #prefix=''
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')



    trainer = pl.Trainer(logger=logger,
                         callbacks=[lr_monitor],
                         log_every_n_steps=2,
                         precision=32,
                         checkpoint_callback=checkpoint_callback,
                         gpus=opt.gpus,
                         num_sanity_val_steps=2,
                         accumulate_grad_batches=opt.virtualBatchSize,
                         max_epochs=opt.nEpochs,
                         sync_batchnorm=False,
                         weights_summary='full',progress_bar_refresh_rate=0)

    optimizer = torch.optim.SGD(arch.parameters(), lr=1e-2,momentum=0.9, weight_decay=1e-6)
    schedulers = {'scheduler': lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[50,80],gamma=0.2),
                  'name': 'step_scheduler',
                  'interval': 'epoch',
                  }

    checkpoint = torch.load('/home/dlachinov/cardiospike/models/001/001_0/epoch=95-step=136511.ckpt')

    compiled_model = pl_model_wrapper.Model(model=arch,
                                            losses=criterion,
                                            metrics=metrics_val,
                                            metametrics=meta_metric_val,
                                            optim=([optimizer],[schedulers]))

    compiled_model.load_state_dict(checkpoint['state_dict'],strict=False)
    loader_helper.freeze(compiled_model.model.encoder_convs)


    trainer.fit(compiled_model,
                train_dataloader=training_data_loader,
                val_dataloaders=evaluation_data_loader)



if __name__ == "__main__":
    opt = parse()

    with open(opt.cv_configuration) as json_file:
        cv_config = json.load(json_file)

    split_list = range(cv_config['n_splits'])

    if not opt.split is None:
        split_list = opt.split

    for s in split_list:
        split_opt = copy.deepcopy(opt)

        f = 'split_'+str(s)+'.p'

        model_path = os.path.join(split_opt.models_path, split_opt.name)
        #shutil.rmtree(split_opt.models_path,ignore_errors=True)
        os.makedirs(model_path, exist_ok=True)

        split_opt.name = opt.name + '_' + str(s)
        split_opt.models_path = model_path

        print('running {} out of {}'.format(s, len(split_list)))
        main(split_opt, cv_config, s)

