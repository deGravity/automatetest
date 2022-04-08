import pytorch_lightning as pl
from automate import SBGCN, LinearBlock, sbgcn
import torch
from typing import Any, List, Optional, Tuple, Union
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torchmetrics
#from torch_geometric.nn import global_max_pool
from .pointnet_encoder import PointNetEncoder
from .mate_predictor_base import MatePredictorBase
#from torch_geometric.nn import GATv2Conv
from .assembly_conv import AssemblyNet


class MatePredictor(MatePredictorBase):
    def __init__(
            self,
            use_sbgcn: bool = True,
            sbgcn_size: int = 64,
            linear_sizes: List[int] = [512, 512],
            n_samples: int = 10,
            f_in: int = 60,
            l_in: int = 38,
            e_in: int = 68,
            v_in: int = 3,
            motion_pointnet: bool = False,
            point_features: int = 7,
            pointnet_size: int = 1024,
            use_uvnet: bool = False,
            crv_emb_dim: int = 64,
            srf_emb_dim: int = 64,
            #num_points: int = 100,
            log_points: bool = False,
            pool_features: bool = False,
            assembly_conv: bool = False,
            assembly_conv_layers: int = 1
        ):
        super().__init__()
        self.log_points = log_points
        self.use_sbgcn = use_sbgcn
        self.sbgcn_size = sbgcn_size
        self.linear_sizes = linear_sizes
        self.f_in = f_in
        self.l_in = l_in
        self.e_in = e_in
        self.v_in = v_in
        self.motion_pointnet = motion_pointnet
        self.point_features = point_features
        self.pointnet_size = pointnet_size
        self.use_uvnet = use_uvnet
        self.crv_emb_dim = crv_emb_dim
        self.srf_emb_dim = srf_emb_dim
        self.pool_features = pool_features
        self.assembly_conv = assembly_conv
        self.assembly_conv_layers = assembly_conv_layers


        #self.num_points = num_points
        out_size = 0
        if self.use_sbgcn:
            self.sbgcn = SBGCN(self.f_in, self.l_in, self.e_in, self.v_in, self.sbgcn_size, 0, use_uvnet_features=self.use_uvnet, crv_emb_dim=self.crv_emb_dim, srf_emb_dim=self.srf_emb_dim)
            out_size += self.sbgcn_size if self.pool_features else self.sbgcn_size * 2
        if self.motion_pointnet:
            self.pointnet_encoder = PointNetEncoder(K=point_features, layers=(64, 64, 64, 128, pointnet_size))
            out_size += self.pointnet_size * 5
        if self.assembly_conv:
            self.gcns = ModuleList([AssemblyNet(sbgcn_size, use_edge_feats=False) for i in range(self.assembly_conv_layers)])

        self.lin = LinearBlock(out_size, *self.linear_sizes, 4, last_linear=True)
        self.loss = torch.nn.CrossEntropyLoss() #TODO: weighting

        self.save_hyperparameters()

    
    def forward(self, graph):
        
        pair_feats = torch.zeros((graph.part_edges.shape[1], 0)).type_as(graph.F)

        if self.use_sbgcn:
            x_t, x_p, _, _, _, _ = self.sbgcn(graph)
            feats_l = x_p[graph.part_edges[0]]
            feats_r = x_p[graph.part_edges[1]]

            if self.assembly_conv:
                for gcn in self.gcns:
                    x_p = gcn(x_p, graph.part_edges)

            if self.pool_features:
                pair_feats = torch.maximum(feats_l, feats_r)
            else:
                pair_feats = torch.cat([feats_l, feats_r], dim=1)

        if self.motion_pointnet:
            _, pointnet_feats = self.pointnet_encoder(graph.motion_points)
            pair_feats = torch.cat([pair_feats, pointnet_feats.reshape(pair_feats.shape[0], -1)], dim=1)

        preds = self.lin(pair_feats)
        return preds

    
    def training_step(self, data, batch_idx):
        target = data.mate_labels

        preds = self(data)
        error = self.loss(preds, target)
        self.log('train_loss/step', error, on_step=True, on_epoch=False)
        self.log('train_loss/epoch', error, on_step=False, on_epoch=True)
        return error

    def validation_step(self, data, batch_idx):
        if batch_idx < 10 and self.log_points and self.motion_pointnet:
            
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