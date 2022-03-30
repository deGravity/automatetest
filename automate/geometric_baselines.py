import pytorch_lightning as pl
from automate import LinearBlock
import torch
from typing import Any, List, Optional, Tuple, Union
from torch.nn import Module
import torch.nn.functional as F
import torchmetrics
#from torch_geometric.nn import global_max_pool
from .pointnet_encoder import PointNetEncoder
from .mate_predictor_base import MatePredictorBase


class PointnetBaseline(MatePredictorBase):
    def __init__(
            self,
            linear_sizes: List[int] = [512, 512],
            point_features: int = 6,
            pointnet_size: int = 1024,
            num_points: int = 100,
            log_points: bool = False
        ):
        super().__init__()

        self.log_points = log_points
        self.num_points = num_points
        self.point_features = point_features
        out_size = 0
        
        self.pointnet_encoder = PointNetEncoder(K=point_features, layers=(64, 64, 64, 128, pointnet_size))
        out_size += 2 * pointnet_size

        self.lin = LinearBlock(out_size, *linear_sizes, 4, last_linear=True)
        self.loss = torch.nn.CrossEntropyLoss() #TODO: weighting

    
    def forward(self, graph):

        _, pointnet_feats = self.pointnet_encoder(graph.pcs)
        pointnet_feats = pointnet_feats.reshape(graph.pcs.shape[0], -1)
        preds = self.lin(pointnet_feats)
        return preds

    
    def training_step(self, data, batch_idx):
        target = data.mate_labels

        preds = self(data)
        error = self.loss(preds, target)
        self.log('train_loss/step', error, on_step=True, on_epoch=False)
        self.log('train_loss/epoch', error, on_step=False, on_epoch=True)
        return error

    def validation_step(self, data, batch_idx):
        if batch_idx < 10 and self.log_points:
            
            self.logger.experiment.add_mesh(f'mesh_vis_{data.ass.item()}', vertices = data.V.unsqueeze(0), faces = data.F.T.unsqueeze(0))

            pcs = data.pcs
            vertices = pcs[:,:,:3]
            col = torch.zeros_like(vertices)
            col[:,:self.num_points,0] = 255
            col[:,self.num_points:,2] = 255
            
            self.logger.experiment.add_mesh(f'points_vis_{data.ass.item()}', vertices=vertices, colors=col)
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