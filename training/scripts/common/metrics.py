import torch
import numpy as np
from torchmetrics import F1

class Metrics(object):
    def __init__(self):
        self.accumulator = []

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        return 0

    def update(self, ground, predict):
        result = self.calculate_batch(ground,predict)
        self.accumulator.extend(result.tolist())

    def get(self):
        return np.nanmean(self.accumulator)

    def reset(self):
        self.accumulator = []

class RMSE(Metrics):
    def __init__(self, output_key=0, target_key=0):
        super(RMSE, self).__init__()
        self.output_key=output_key
        self.target_key=target_key

    def calculate_batch(self, ground:dict, predict:dict) -> np.ndarray:
        pred = predict[self.output_key].detach()
        gr = ground[self.target_key].detach()

        assert (gr.shape == pred.shape)

        N = gr.shape[0]

        pred = pred.view(N,-1)
        gr = gr.view(N,-1)

        result = (pred - gr) ** 2

        return result.mean(dim=1).cpu().numpy()

    def get(self):
        return np.sqrt(np.nanmean(self.accumulator))

def print_metrics(writer, name, metric, prefix, epoch):
    if isinstance(metric.get(), np.ndarray):
        for i in range(metric.get().shape[0]):
            writer.add_scalar(prefix + name+str(i), metric.get()[i], epoch)
    else:
        writer.add_scalar(prefix + name, metric.get(), epoch)

    print('Epoch %d, %s %s %s' % (epoch, prefix, name, metric.get()))

class F1micro(Metrics):
    def __init__(self, output_key=0, target_key=0, slice=0):
        super(F1micro, self).__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice

        self.F1 = F1()
        self.scores = []

    def reset(self):
        self.scores = []

    def update(self, ground: dict, predict: dict):
        pred = predict[self.output_key][:, self.slice].detach().cpu()
        gr = ground[self.target_key][:, self.slice].detach().cpu().int()

        self.scores.append(self.F1(pred.flatten(),gr.flatten()>0.5))

    def get(self):

        return np.mean(self.scores)


class F1global(Metrics):
    def __init__(self, output_key=0, target_key=0, slice=0):
        super(F1global, self).__init__()
        self.output_key=output_key
        self.target_key=target_key
        self.slice=slice

        self.F1 = F1()
        self.preds = []
        self.gts = []

    def reset(self):
        self.preds = []
        self.gts = []

    def update(self, ground: dict, predict: dict):
        pred = predict[self.output_key][:, self.slice].detach().cpu()
        gr = ground[self.target_key][:, self.slice].detach().cpu().int()

        self.preds.append(pred.flatten())
        self.gts.append(gr.flatten())

    def get(self):

        return self.F1(torch.cat(self.preds,dim=0),torch.cat(self.gts,dim=0)>0.5)