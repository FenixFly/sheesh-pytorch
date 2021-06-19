# Trainig scripts for Cardiospike project

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

Navigate to one of the `001c`,`002c`, `003c` folders. They have identical structure, where `main.py` is main 


### Validating the models and the ensamble

### Running inference and ensembling
