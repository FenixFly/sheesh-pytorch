import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import re

from helpers import postprocess

if __name__ == '__main__':
    results={   '001c11':'/home/dlachinov/cardiospike/scripts/001c/prediction_001c11.csv',
                '002c1':'/home/dlachinov/cardiospike/scripts/002c/prediction_002c1.csv',
                '003c3':'/home/dlachinov/cardiospike/scripts/003c/prediction_003c3.csv',
                }

    #rx = re.compile('result_(.+)_split_(\d+)')

    rx = re.compile('result_(.+)_split_0')

    threshold = 0.53
    stack = []
    cvr = None
    for model, path in results.items():
        cvr = pd.read_csv(path)

        for col in cvr.columns:
            if rx.match(col):
                stack.append(cvr[col])


    predict = sum(stack)/len(stack)
    print('stack len', len(stack))

    df_pred = cvr[['id','time','x']].copy()
    df_pred['y'] = (predict > threshold).astype(int)#postprocess.postprocess((predict > threshold).astype(int).to_numpy())

    df_pred.to_csv('submission9.csv',index=False)



