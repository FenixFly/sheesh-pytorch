# Training scripts for Cardiospike project

## Structure of the README

- [Setting up the folder structure](#Proper-folder-structure)
- [Making cross-validation splits](#Making-cross-validation-splits)
- [Training the models](#Training-the-models)
- [Validating the models and the ensamble](#Validating-the-models-and-the-ensamble)
- [Running inference and ensembling](#Running-inference-and-ensembling)

### Proper folder structure
First, ensure that the project has the following folder structure:

```
├── data
├── models
├── scripts
│   ├── 001c
│   ├── 002c
│   ├── 003c
│   ├── common
│   └── helpers
└── splits
    ├── cardiospike
    ├── WFDB_ChapmanShaoxing
    ├── WFDB_CPSC2018
    ├── WFDB_CPSC2018_2
    ├── WFDB_Ga
    ├── WFDB_Ningbo
    ├── WFDB_PTB
    ├── WFDB_PTBXL
    └── WFDB_StPetersburg
```

`data` - all data are stored here. Place `train.csv` here.
`models` - will contain logs and checkpoints
`scripts` - folders with numbers are three different models
`splits` - folder with splits for cross-validation

### Making cross-validation splits (optional)
Before running the training you need to generate CV split and configureation file. This, however, optional. You can reuse existing folds and `config_cardiospike.cfg` from the training root. Make sure that the config file has correct relative paths

For making the splits, open `helpers/make_splits.py`, edit IO paths and run. No command line args are required

### Training the models

Navigate to one of the `001c`,`002c`, `003c` folders. They have identical structure, where `main.py` is the main file, `model.py`contains the model definition, `validate.py` running inference on cross-validation (metrics are reported by the script `helpers/analyze_stack_results.py`), `infer.py` performs inference.

For training run `main.py`.
```
main.py --batchSize 16 --virtualBatchSize 1 --nEpochs 100 --name 001c1 --threads 16 --root_dir <root dir for training> --cv_configuration config_cardispike.cfg --models_path /home/dlachinov/cardiospike/models --gpus 1

--batchSize and --virtualBatchSize controll size of the batch
--nEpochs is the number of the training epoches
--name is a name of this configuration
--threads n threads to be used by dataloader
--root_dir path to root of the training ex. sheesh-pytorch/training
--cv_configuration path to cfg file
--gpus number of gpus to use
```
You can track progress with tensorboard `models/00c1/00c1_0/logs`, where `00c1_0` is a split.


Train each of 001c, 002c, 003c models

### Validating the models and the ensamble
After the training, you can run cross-validation. It is decoupled into 2 parts: running predictions on validation folds `validate.py` and aggregating and validating all models with `helpers/analyze_stack_results.py`

For each individual model first run:
```
validate.py --name 004c1 --root_dir<root dir for training> --cv_configuration config_cardispike.cfg --models_path models/004c1
parameters are the same as in training
make sure that --models_path is correct and points to the folder with 10 subfolder (number of splits)
```
After that `cv_result.csv` will be generated

Navigate to helpers and edit `analyze_stack_results.py`. Make sure that dictionary cv_results has correct path. Then run the script. No arguments required.

### Running inference and ensembling

Inference scheme is similar to validation.
First, in each model folder run `infer.py`

```
infer.py --data_path ./test.csv --name 003c2 --models_path ./models
```

File named `prediction_<>.csv` will be created in the same folder.

Now navigate to helpers and run `stack_predicts.py`. Make Sure that the results dict has correct paths.
After running the script, submission file should appear in the same folder.

