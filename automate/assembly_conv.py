import torch
from .sbgcn import LinearBlock
import torch_scatter

class ResMRConvWithEdgeFeats(torch.nn.Module):
    def __init__(self, width, num_features, batch_norm=False, dropout=0.0):
        super().__init__()
        self.mlp = LinearBlock(2*width, width, batch_norm=batch_norm, dropout=dropout)
        self.edge_mlp = LinearBlock(width + num_features, width, batch_norm=batch_norm, dropout=dropout)
    
    def forward(self, x, e, e_feat):
        diffs = torch.index_select(x, 0, e[1]) - torch.index_select(x, 0, e[0])
        e_feat = diffs + self.edge_mlp(torch.cat([diffs, e_feat], dim=1))
        maxes, _ = torch_scatter.scatter_max(
            e_feat, 
            e[1], 
            dim=0, 
            dim_size=x.shape[0]
        )
        return x + self.mlp(torch.cat([x, maxes], dim=1))

class ResMRConv(torch.nn.Module):
    def __init__(self, width, batch_norm=False, dropout=0.0):
        super().__init__()
        self.mlp = LinearBlock(2*width, width, batch_norm=batch_norm, dropout=dropout)
    
    def forward(self, x, e):
        diffs = torch.index_select(x, 0, e[1]) - torch.index_select(x, 0, e[0])
        maxes, _ = torch_scatter.scatter_max(
            diffs, 
            e[1], 
            dim=0, 
            dim_size=x.shape[0]
        )
        return x + self.mlp(torch.cat([x, maxes], dim=1))

class AssemblyNet(torch.nn.Module):
    def __init__(self, width, batch_norm=False, dropout=0.0, use_edge_feats=True, num_edge_feats=2, close_pairs_only=False):
        super().__init__()
        self.use_edge_feats = use_edge_feats
        self.close_pairs_only = close_pairs_only
        if use_edge_feats:
            self.P2P = ResMRConvWithEdgeFeats(width, num_edge_feats, batch_norm=batch_norm, dropout=dropout)
        else:
            self.P2P = ResMRConv(width, batch_norm=batch_norm, dropout=dropout)
    
    def forward(self, x_p, edge_indices, edge_feats=None):
        part_edges_sym = torch.cat([edge_indices, torch.flip(edge_indices, dims=(0,))], dim=1)
        
        if self.use_edge_feats:
            part_feats_sym = torch.cat([edge_feats, edge_feats], dim=0)
            x_p = self.P2P(x_p, part_edges_sym, part_feats_sym)
        else:
            x_p = self.P2P(x_p, part_edges_sym)

        return x_p