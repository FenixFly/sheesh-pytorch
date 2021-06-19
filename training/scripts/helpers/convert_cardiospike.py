import pandas as pd
import os

if __name__ == '__main__':

    input_datasets = '/home/dlachinov/cardiospike/data/cardiospike/train.csv'
    output_datasets = '/home/dlachinov/cardiospike/filestorage'

    dataset = pd.read_csv(input_datasets)

    ids = dataset['id'].unique()

    metadata_list = []
    for id in ids:

        signal = dataset[dataset['id']==id].sort_values(by='time',ascending=True)

        save_path = os.path.join(output_datasets,str(id))
        os.makedirs(save_path,exist_ok=True)

        signal[['time','x','y']].to_csv(os.path.join(save_path,'result.csv'),index=False)
        signal[['time','x']].to_csv(os.path.join(save_path,'original.csv'),index=False)

        record  = {}
        record['name'] = id
        record['dataset'] = 'cardispike'
        record['peaks_n'] = signal.shape[0]



        metadata_list.append(record)


    df = pd.DataFrame(metadata_list)
    df.to_csv(os.path.join(output_datasets,'metadata.csv'))