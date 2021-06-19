import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
from helpers import postprocess

def get_stat(gt, cvr, threshold):
    f1_all_samples = f1_score(gt['y'], cvr['y_pred']>threshold)

    ids = gt['id'].unique()

    results = []
    for id in ids:

        gt_sample = gt[gt['id']==id]
        cvr_sample = cvr[cvr['id']==id]
        split = cvr_sample.iloc[0]['split']
        r={'id':id,
           'split':split,
           'f1':f1_score(gt_sample['y'],cvr_sample['y_pred']>threshold)}

        results.append(r)

    results = pd.DataFrame(results)
    return  f1_all_samples, results['f1'].mean(), results.groupby(by=['split'])['f1'].mean()


if __name__ == '__main__':
    cv_results={#'001c1':'/home/dlachinov/cardiospike/scripts/001c/cv_results.csv',
                #'001c2':'/home/dlachinov/cardiospike/scripts/001c/cv_results_c2.csv',
                #'001c3':'/home/dlachinov/cardiospike/scripts/001c/cv_results_c3.csv',
                #'001c9':'/home/dlachinov/cardiospike/scripts/001c/cv_results_c9.csv',
                #'001c10':'/home/dlachinov/cardiospike/scripts/001c/cv_results_c10.csv',
                '001c11':'/home/dlachinov/cardiospike/scripts/001c/cv_results_c11.csv',
                #'001c12':'/home/dlachinov/cardiospike/scripts/001c/cv_results_c12.csv',
                '002c1':'/home/dlachinov/cardiospike/scripts/002c/cv_results_c1.csv',
                #'002c2':'/home/dlachinov/cardiospike/scripts/002c/cv_results_c2.csv',
                #'002c3':'/home/dlachinov/cardiospike/scripts/002c/cv_results_c3.csv',
                #'003c1':'/home/dlachinov/cardiospike/scripts/003c/cv_results_c1.csv',
                #'003c2':'/home/dlachinov/cardiospike/scripts/003c/cv_results_c2.csv',
                '003c3':'/home/dlachinov/cardiospike/scripts/003c/cv_results_c3.csv',
                #'004c1':'/home/dlachinov/cardiospike/scripts/003c/cv_results_c1.csv',
                }

    gt = pd.read_csv('/home/dlachinov/cardiospike/data/cardiospike/train.csv').sort_values(by=['id','time'],ascending=True).set_index(keys=['id','time'],drop=False)


    threshold = 0.5
    stack = []
    for model, path in cv_results.items():
        cvr = pd.read_csv(path,index_col=0).sort_values(by=['id','time'],ascending=True).set_index(keys=['id','time'],drop=False)
        stack.append(cvr)

        print('All cases are in table: ',gt.index.equals(cvr.index))

        f1_all_samples, f1_per_id, group_by_split = get_stat(gt, cvr, threshold)
        print(model, 'f1_all_samples', f1_all_samples)
        print(model, 'mean sample f1', f1_per_id)
        print(model, 'mean sample f1', group_by_split)

    # no postprocess            0.8656
    # delete <6                 0.8657
    # dilate elem 3             0.8640
    # delete <3 dilate 3<=x<6   0.8672
    # delete <4 dilate 4<=x<6   0.8681

    cvr = stack[0]

    cvr['y_pred'] = sum([c['y_pred'] for c in stack])/len(stack)

    binarized = (cvr['y_pred'].to_numpy() > 0.5).astype(np.int32)
    binarized = postprocess.postprocess(binarized)
    print('postprocessed f1', f1_score(gt['y'], binarized > threshold))

    f1_all_samples, f1_per_id, group_by_split = get_stat(gt, cvr, 0.5)

    print('ensable', 'f1_all_samples', f1_all_samples)
    print('ensable', 'mean sample f1', f1_per_id)
    print('ensable', 'mean sample f1', group_by_split)


    thresholds = np.linspace(0,1,num=50)
    f1s = []

    for t in thresholds:
        f1_all_samples, f1_per_id, group_by_split = get_stat(gt, cvr, t)
        f1s.append(f1_all_samples)

    plt.plot(thresholds,f1s)
    plt.show()

    idx = np.argmax(f1s).item()
    print('best f1 ', f1s[idx], 'threshold ', thresholds[idx])



