from zipfile import ZipFile
import torch
from torch.utils.data import Dataset
import json
import os
import random
import pickle
from pspy import Part, ImplicitPart, PartOptions
from automate import part_to_graph, PartFeatures, implicit_part_to_data
from functools import cache

'''
Dataset structure:

Low Level: Index + Zip or Directory -> Part Object / PyGeo + Labels
Mid Level: Caching: In Memory and / or On Disk
High Level: Subsetting - let us change subsets without losing cache

Considerations:

Mode: train and test have unique index set numbers, but only one should
be subset or split for validation.

Cache-Retention: It would be great to keep the in-memory cache between
experiments, but this would require keeping the data loaders alive while
somehow changing their underlying data sets.

Labels: mfcad and fusion360seg have similar label structures. We could
do the same for automate, or just load the labels into the index jsons
on everything once and have a more consistent labeling structure. Labels
also make validation sampling trickier since we want to do stratified
sampling in these cases.

'''

class CachedDataset(Dataset):
    def __init__(self, ds, cache_dir = None, memcache = False):
        super().__init__()
        self.ds = ds
        self.memcache = memcache
        self.cache_dir = cache_dir
        if hasattr(ds, 'labels'):
            self.labels = ds.labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if (res := self.get_memcached(idx)):
            pass
        elif (res := self.get_filecached(idx)):
            pass
        else:
            res = self.ds[idx]
        self.cache_file(res, idx)
        self.cache_mem(res, idx)
        return res

    def get_memcached(self, idx):
        if self.memcache:
            if not self.cache:
                self.cache = dict()
            return self.cache.get(idx, None)
        return None
    
    def get_filecached(self, idx):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f'{idx}.pt')
            if os.path.exists(cache_path):
                return torch.load(cache_path)
        return None
    
    def cache_file(self, idx):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f'{idx}.pt')
            if not os.path.exists(cache_path):
                cache_dir = os.path.dirname(cache_path)
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(cache_path)
        

class PartDataset(Dataset):
    def __init__(
        self, 
        index_path: str, 
        data_path: str,  
        mode: str = 'train',
        part_options: dict = {},
        implicit: bool = True,
        feature_options: PartFeatures = PartFeatures(),
        implicit_options: dict = {}
    ):
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        self.template = index['template']
        ids = index.get(mode, index['train'])
        self.ids = ids
        if 'train_labels' in index:
            self.labels = index.get(f'{mode}_labels', index['train_labels'])
            assert(len(ids) == len(self.labels))
        self.data_path = data_path
        self.part_options = part_options
        self.implicit = implicit
        self.feature_options = feature_options
        self.implicit_options = implicit_options

    def get_options_struct(self):
        options = PartOptions()
        for k,v in self.part_options.items():
            if hasattr(options, k):
                setattr(options, k, v)
        return options

    def __len__(self):
        pass

    def __getitem__(self, idx):
        model_path = self.template.format(*self.ids[idx])
        if self.data_path.endswith('.zip'): # Read from Zip
            if not self.datafile:
                self.datafile = ZipFile(self.data_path)
            with self.datafile.open(model_path, 'r') as f:
                model_data = f.read()
                if self.implicit:
                    model = ImplicitPart(model_data, **self.part_options)
                else:
                    model = Part(model_data, self.get_options_struct())
        else: # Read from directory
            model_path = os.path.join(self.data_path, model_path)
            if self.implicit:
                model = ImplicitPart(model_path, **self.part_options)
            else:
                model = Part(model_path, self.get_options_struct())
        
        if self.implicit:
            graph = implicit_part_to_data(model, **self.implicit_options)
        else:
            graph = part_to_graph(model, self.feature_options)
        
        if hasattr(self, 'labels'):
            graph.labels = torch.tensor(self.labels[idx]).long()

        return graph

