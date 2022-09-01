import torch_geometric as tg
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import random
import torch
import json
from tqdm import tqdm

def create_subset(data, seed, size):
    random.seed(seed)
    return random.sample(data, size)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = tg.nn.GCNConv(in_channels, 2*out_channels)
        self.conv2 = tg.nn.GCNConv(2*out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class CodeGraphAutoencoder(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = tg.nn.GAE(GCNEncoder(64,16))
    def forward(self, batch):
        return self.model.encode(batch.x, batch.edge_index)
    def training_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.model.recon_loss(z, batch.edge_index)
        self.log('train_loss',loss, on_step=True,on_epoch=True,batch_size=z.shape[0])
        return loss
    def validation_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.model.recon_loss(z, batch.edge_index)
        self.log('val_loss',loss,on_step=True,on_epoch=True,batch_size=z.shape[0])
    def test_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.model.recon_loss(z, batch.edge_index)
        self.log('test_loss',loss,on_step=False,on_epoch=True,batch_size=z.shape[0])
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

from automate import LinearBlock, BipartiteResMRConv
from torch.nn import ModuleList
from torchmetrics import Accuracy
from torch.nn.functional import cross_entropy
class CodePredictor(pl.LightningModule):
    def __init__(self, in_channels, out_channels, mlp_layers, mp_layers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_layers = mlp_layers
        self.mp_layers = mp_layers
        
        self.mp = ModuleList([BipartiteResMRConv(in_channels) for _ in range(mp_layers)])
        self.mlp = LinearBlock(*([in_channels]*mlp_layers), out_channels, last_linear=True)
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        
    def forward(self, data):
        x = data.x#torch.cat([data.x,data.z],dim=1)
        for mp in self.mp:
            x = mp(x,x,data.edge_index)
        x = self.mlp(x)
        return x
    
    def training_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        target = batch.y.reshape(-1)
        loss = cross_entropy(scores, target)
        self.train_acc(preds, target)
        batch_size = len(target)
        self.log('train_loss',loss,on_epoch=True,on_step=True,batch_size=batch_size)
        self.log('train_acc',self.train_acc,on_epoch=True,on_step=True,batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        target = batch.y.reshape(-1)
        loss = cross_entropy(scores, target)
        self.val_acc(preds, target)
        batch_size = len(target)
        self.log('val_loss',loss,on_epoch=True,on_step=True,batch_size=batch_size)
        self.log('val_acc',self.val_acc,on_epoch=True,on_step=True,batch_size=batch_size)
    
    def test_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        target = batch.y.reshape(-1)
        loss = cross_entropy(scores, target)
        self.test_acc(preds, target)
        batch_size = len(target)
        self.log('test_loss',loss,on_epoch=True,on_step=False,batch_size=batch_size)
        self.log('test_acc',self.test_acc,on_epoch=True,on_step=False,batch_size=batch_size)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, index, data, mode='train', val_frac=.2,seed=42,undirected=True,train_size=None):
        index
        keyset = index['test']
        if mode in ['train', 'validate']:
            keyset = index['train']
            if train_size:
                keyset = create_subset(keyset, seed, train_size)
            train_keys, val_keys = train_test_split(keyset, test_size=val_frac, random_state=seed)
            keyset = train_keys if mode == 'train' else val_keys
        self.data = {i:tg.data.Data(**data[index['template'].format(*key)]) for i,key in enumerate(keyset)}
        if undirected:
            for k,v in self.data.items():
                v.edge_index = torch.cat([v.edge_index, v.edge_index[[1,0]]],dim=1)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class DictDatamodule(pl.LightningDataModule):
    def __init__(self, index, data, val_frac=0.2, seed=42, batch_size=32,train_size=None):
        super().__init__()
        self.val_frac = val_frac
        self.seed = seed
        self.ds_train = DictDataset(index, data, 'train', val_frac, seed, True, train_size)
        self.ds_val = DictDataset(index, data, 'validate', val_frac, seed, True, train_size)
        self.ds_test = DictDataset(index, data, 'test')
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=min(len(self.ds_train), self.batch_size), 
            shuffle=True, 
            num_workers=8, 
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val, 
            batch_size=min(len(self.ds_val),self.batch_size), 
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=1, shuffle=False)