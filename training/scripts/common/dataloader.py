import torch.utils.data as data
import pandas as pd


class CardioSpikeDataset(data.Dataset):
    def __init__(self, path, patients:list=None, multiplier=1, transforms=None, validation=False):
        super(CardioSpikeDataset, self).__init__()
        self.path = path
        self.multiplier = multiplier
        self.transforms = transforms
        self.validation = validation

        self.dataset = pd.read_csv(path)

        if patients is None:
            self.patients = self.dataset['id'].unique().tolist()
        else:
            self.patients = patients

        self.patients = [int(float(p)) for p in self.patients]
        self.dataset = self._make_dataset(self.patients)
        self.real_length = len(self.patients)

    def _make_dataset(self,patients):
        index = self.dataset['id'].isin(patients)
        return self.dataset[index]


    def __load(self, index):

        id = self.patients[index]

        data = self.dataset.loc[self.dataset['id']==id,:].sort_values(by='time', axis=0, ascending=True)

        record = {}

        record['id'] = id
        record['signal'] = data['x'].to_numpy()[None]
        record['label'] = data['y'].to_numpy()[None]
        record['time'] = data['time'].to_numpy()[None]
        record['x'] = data['x'].to_numpy()[None]
        return record

    def __getitem__(self, index):
        index = index % self.real_length

        record = self.__load(index)

        if self.transforms is not None:
            record = self.transforms(record)

        return record

    def __len__(self):
        return int(self.multiplier * self.real_length)