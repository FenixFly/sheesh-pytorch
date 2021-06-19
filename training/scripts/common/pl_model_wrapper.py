import pytorch_lightning as pl
import loader_helper

class Model(pl.LightningModule):

    def __init__(self,model, losses, metrics, metametrics, optim):
        super().__init__()
        self.model = model
        self.loss = losses
        self.metrics = metrics
        self.metametrics = metametrics
        self.optim = optim



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        res = self.model(batch)
        loss, values = self.loss(batch, res)

        for k in values:
            self.log('Training/'+str(k), values[k], on_step=True,on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        res = self.model(batch)

        for k in self.metrics:
            self.metrics[k].update(batch,res)

    def validation_epoch_end(self, validation_step_outputs):

        metric_results = {k:self.metrics[k].get() for k in self.metrics}

        for k in self.metrics:
            self.log('Validation_'+str(k), metric_results[k], on_epoch=True)
            self.metrics[k].reset()

        if self.metametrics is not None:
            for k in self.metametrics:
                self.log('Metametrics/' + str(k), self.metametrics[k].get(metric_results), on_epoch=True)


    def configure_optimizers(self):
        return self.optim