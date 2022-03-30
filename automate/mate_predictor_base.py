import pytorch_lightning as pl
from .plot_confusion_matrix import plot_confusion_matrix
import torchmetrics
import torch

sub_mate_types = [
    'FASTENED',
    'SLIDER',
    'REVOLUTE',
    'CYLINDRICAL'
]

class MatePredictorBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=4, compute_on_step=False)
        binary_metrics = torchmetrics.MetricCollection([
               torchmetrics.F1(compute_on_step=False),
               torchmetrics.Precision(compute_on_step=False),
               torchmetrics.Recall(compute_on_step=False),
               torchmetrics.Accuracy(compute_on_step=False)])

        self.type_accuracy = torchmetrics.Accuracy(compute_on_step=False, threshold=0)
        self.fasten_stats = binary_metrics.clone(prefix='val_individual/fasten_')
        self.slider_stats = binary_metrics.clone(prefix='val_individual/slider_')
        self.revolute_stats = binary_metrics.clone(prefix='val_individual/revolute_')
        self.cylindrical_stats = binary_metrics.clone(prefix='val_individual/cylindrical_')
        self.sliding_stats = binary_metrics.clone(prefix='val_sliding/')
        self.rotating_stats = binary_metrics.clone(prefix='val_rotating/')
        self.axis_stats = binary_metrics.clone(prefix='val_axis/')


    def log_confusion_matrix(self, mode):
        cm = self.confusion_matrix.compute()
        cm_fig_count = plot_confusion_matrix(cm, sub_mate_types, (-1, 'Count'))
        cm_fig_precision = plot_confusion_matrix(cm, sub_mate_types, (0, 'Precision'))
        cm_fig_recall = plot_confusion_matrix(cm, sub_mate_types, (1, 'Recall'))
        self.confusion_matrix.reset()
        self.logger.experiment.add_figure(mode + '_confusion_matrix/Precision', cm_fig_precision, self.current_epoch)
        self.logger.experiment.add_figure(mode + '_confusion_matrix/Recall', cm_fig_recall, self.current_epoch)
        self.logger.experiment.add_figure(mode + '_confusion_matrix/Count', cm_fig_count, self.current_epoch)


    def validation_epoch_end(self, outputs):
        self.log_confusion_matrix('val')
    

    def test_epoch_end(self, outputs):
        self.log_confusion_matrix('test')
    

    def log_metrics(self, target, preds, mode):
        self.confusion_matrix(preds, target)
        self.type_accuracy(preds, target)
        self.log(mode + '_type_accuracy', self.type_accuracy, on_step=False, on_epoch=True)

    
    def get_callbacks(self):
        callbacks = [
            pl.callbacks.ModelCheckpoint(monitor="val_type_accuracy", save_top_k=4, filename="{epoch}-{val_type_accuracy:.6f}", mode="max"),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=4, filename="{epoch}-{val_loss:.6f}", mode="min"),
            pl.callbacks.ModelCheckpoint(save_top_k=-1, every_n_epochs=5),
            pl.callbacks.ModelCheckpoint(save_last=True),
        ]
        return callbacks
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer