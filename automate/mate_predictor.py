import pytorch_lightning as pl
from automate import SBGCN, LinearBlock, part_to_graph
import torch
from typing import Any, List, Optional, Tuple, Union
from torch.nn import Module
import torch.nn.functional as F
import torchmetrics
#from torch_geometric.nn import global_max_pool
from .plot_confusion_matrix import plot_confusion_matrix

sub_mate_types = [
    'FASTENED',
    'SLIDER',
    'REVOLUTE',
    'CYLINDRICAL'
]

class PointNetEncoder(Module):
    def __init__(self, K=3, layers=(64, 64, 64, 128, 1024)):
        super().__init__()
        self.encode = LinearBlock(K, *layers)
        self.K = K
    def forward(self, pc):
        pc2 = pc.reshape(-1, self.K)
        x = self.encode(pc2)
        x = x.reshape(pc.shape[0], pc.shape[1], pc.shape[2], -1)
        x_p = torch.max(x, dim=2)[0]
        return x, x_p

class MatePredictor(pl.LightningModule):
    def __init__(
            self,
            sbgcn_size: int = 64,
            linear_sizes: List[int] = [512, 512],
            n_samples: int = 10,
            f_in: int = 60,
            l_in: int = 38,
            e_in: int = 68,
            v_in: int = 3,
            pointnet: bool = False,
            point_features: int = 7,
            pointnet_size: int = 1024,
            #num_points: int = 100,
            log_points: bool = False
        ):
        super().__init__()
        self.log_points = log_points
        #self.num_points = num_points
        self.pointnet = pointnet
        self.point_features = point_features
        self.sbgcn = SBGCN(f_in, l_in, e_in, v_in, sbgcn_size, 0)
        out_size = sbgcn_size * 2
        if self.pointnet:
            self.pointnet_encoder = PointNetEncoder(K=point_features, layers=(64, 64, 64, 128, pointnet_size))
            out_size += pointnet_size * 5

        self.lin = LinearBlock(out_size, *linear_sizes, 4, last_linear=True)
        self.loss = torch.nn.CrossEntropyLoss() #TODO: weighting


        #metrics
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
    
    def forward(self, graph):
        x_t, x_p, _, _, _, _ = self.sbgcn(graph)

        feats_l = x_p[graph.part_edges[0]]
        feats_r = x_p[graph.part_edges[1]]

        pair_feats = torch.cat([feats_l, feats_r], dim=1)

        if self.pointnet:
            _, pointnet_feats = self.pointnet_encoder(graph.motion_points)
            pair_feats = torch.cat([pair_feats, pointnet_feats.reshape(pair_feats.shape[0], -1)], dim=1)

        preds = self.lin(pair_feats)
        return preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, data, batch_idx):
        target = data.mate_labels

        preds = self(data)
        error = self.loss(preds, target)
        self.log('train_loss', error,batch_size=32)
        return error

    def log_metrics(self, target, preds, mode):
        self.confusion_matrix(preds, target)
        self.type_accuracy(preds, target)
        self.log(mode + '_type_accuracy', self.type_accuracy, on_step=False, on_epoch=True)

    def validation_step(self, data, batch_idx):
        if batch_idx < 10 and self.log_points and self.pointnet:
            
            self.logger.experiment.add_mesh(f'mesh_vis_{batch_idx}', vertices = data.V.unsqueeze(0), faces = data.F.T.unsqueeze(0))

            pcs = data.motion_points
            vertices = pcs[:,0,:,:3]
            col = torch.zeros_like(vertices)
            col[:,:100,0] = 255
            col[:,100:,2] = 255
            
            self.logger.experiment.add_mesh(f'points_vis_{batch_idx}', vertices=vertices, colors=col)
            #self.logger.experiment.add_mesh(f'mesh_pair_vis_{batch_idx}', vertices = data.V.unsqueeze(0), faces = data.debug_mesh_pairs[0][0].unsqueeze(0))
            #self.logger.experiment.add_mesh(f'full_mesh_vis_{batch_idx}', vertices=data.pc.unsqueeze(0))

        target = data.mate_labels
        preds = self(data)
        error = self.loss(preds, target)
        self.log('val_loss', error,batch_size=32)
        self.log_metrics(target, preds, 'val')
        #face_classes = data.faces[:,:13].argmax(dim=1).max()
        #face_errors = torch.nn.functional.mse_loss(preds, target, reduction='none')
        #class_errors = scatter_mean(face_errors, face_classes)
        #for i in range(len(class_errors)):
        #    self.log(f'val_loss_class_{i}', class_errors[i])
    

    def test_step(self, data, batch_idx):
        target = data.mate_labels
        preds = self(data)
        error = self.loss(preds, target)
        self.log('test_loss', error,batch_size=32)
        self.log_metrics(target, preds, 'test')
    
    def validation_epoch_end(self, outputs):
        cm = self.confusion_matrix.compute()
        cm_fig_count = plot_confusion_matrix(cm, sub_mate_types, (-1, 'Count'))
        cm_fig_precision = plot_confusion_matrix(cm, sub_mate_types, (0, 'Precision'))
        cm_fig_recall = plot_confusion_matrix(cm, sub_mate_types, (1, 'Recall'))
        self.confusion_matrix.reset()
        self.logger.experiment.add_figure(f'confusion_matrix/Precision', cm_fig_precision, self.current_epoch)
        self.logger.experiment.add_figure(f'confusion_matrix/Recall', cm_fig_recall, self.current_epoch)
        self.logger.experiment.add_figure(f'confusion_matrix/Count', cm_fig_count, self.current_epoch)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)  

    def get_callbacks(self):
        callbacks = [
            pl.callbacks.ModelCheckpoint(save_last=True),
            pl.callbacks.ModelCheckpoint(save_top_k=-1, every_n_epochs=5),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=4, filename="{epoch}-{val_loss:.6f}", mode="min"),
            pl.callbacks.ModelCheckpoint(monitor="val_type_accuracy", save_top_k=4, filename="{epoch}-{val_type_accuracy:.6f}", mode="max")
        ]
        return callbacks
    
    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return pl.utilities.argparse.from_argparse_args(cls, args, **kwargs)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = pl.utilities.argparse.add_argparse_args(cls, parent_parser)
        return parser


if __name__ == '__main__':
    from transforms import *
    from cached_dataset import SavedDataset
    from torch_geometric.data import DataLoader
    transforms = [fix_edge_sets, remap_type_labels]
    dataset = SavedDataset('/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/simple_valid_dataset.txt', '/fast/jamesn8/assembly_data/assembly_torch2_fixsize/new_axes_100groups_and_mate_check/axis_data', prefix_path='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/pspy_batches/batches/', exclusion_file=['/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/assemblies_with_discrepant_mcs.txt', '/fast/jamesn8/assembly_data/assembly_torch2_fixsize/test_add_mate_labels/stats.parquet'], transforms = transforms, debug=True)
    dataloader = DataLoader(dataset, 1)

    model = MatePredictor()

    for i,data in enumerate(dataloader):
        print(i)
        #preds = model(data)
        loss = model.training_step(data, i)
        print(loss)