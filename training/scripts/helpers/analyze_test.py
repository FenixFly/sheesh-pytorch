import pandas as pd
import numpy as np



test = pd.read_csv('/home/dlachinov/cardiospike/data/cardiospike/test.csv')
ids = test['id'].unique()
print('unique ids', len(ids))