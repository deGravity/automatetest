from zipfile import ZipFile
import torch
from torch.utils.data import Dataset
import json
import os
import random
import pickle
from pspy import Part, ImplicitPart, PartOptions
from functools import cache

class CachedDataset(Dataset):
    def __init__(self, ds, cache_dir = None, memcache = False):
        super().__init__()
        self.ds = ds
        self.memcache = memcache
        self.cache_dir = cache_dir

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
            cache_path = os.path.join(self.cache_dir, idx)
            if os.path.exists(cache_path):
                return torch.load(cache_path)
        return None
        

class PartDataset(Dataset):
    def __init__(
        self, 
        index_path: str, 
        data_path: str,  
        mode: str = 'train',
        subset = 0,
        subset_seed = 0,
        val_pct = 0,
        random_validation = False,
        label_type: str = 'txt',
        part_options: dict = {},
        implicit: bool = True
    ):
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        self.template = index['template']
        self.label_template = index.get('label_template', None)
        ids = index.get(mode, index['train'])

        self.label_type = index.get('label_type', label_type)
        if self.label_type == 'txt':
            self.read_labels = lambda f: [x.strip() for x in f.readlines()]
        elif self.label_type == 'pickle':
            self.read_labels = lambda f: pickle.load(f)
        else:
            pass

        # Reduce to a subset
        if subset > 0:
            if subset < 1: # Percentage Subset

                pass
            else: # Fixed Size Subset
                random.seed(subset_seed)
                ids = random.sample(ids, subset)
        
        # Create a validation set
        if random_validation and val_pct > 0 and mode != "train":
            random.shuffle(ids)
        if mode=='train':
            ids = [x for i,x in enumerate(ids) if i % 100 <= (100 - val_pct)]
        elif mode=='validate':
            ids = [x for i,x in enumerate(ids) if i % 100 > (100 - val_pct)]

        self.ids = ids
        self.data_path = data_path
        self.part_options = part_options

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
        label_path = self.label_template.format(*self.ids[idx]) if self.label_template else None
        part_options = self.get_options_struct()
        labels = [self.ids[idx][0]]
        if self.data_path.endswith('.zip'): # Read from Zip
            if not self.datafile:
                self.datafile = ZipFile(self.data_path)
            with self.datafile.open(model_path, 'r') as f:
                model_data = f.read()
                if self.implicit:
                    model = ImplicitPart(model_data, **self.part_options)
                else:
                    model = Part(model_data, part_options)
            if label_path:
                with self.datafile.open(label_path, 'r') as f:
                    labels = self.read_labels(f)
        else: # Read from directory
            model_path = os.path.join(self.data_path, model_path)
            label_path = os.path.join(self.data_path, label_path) if label_path else None
            model = Part(model_path, part_options)
            if label_path:
                with open(label_path, 'r') as f:
                    labels = self.read_labels(f)
        return model, labels

class ImplicitPartDataset(PartDataset):
    def __init__(self, 
        index_path: str, 
        data_path: str,  
        mode: str = 'train',
        subset = 0,
        subset_seed = 0,
        val_pct = 0,
        random_validation = False,
        label_type: str = 'txt',
        part_options: dict = {}
    ):
        super().__init__(
            index_path, data_path, mode, subset, subset_seed, 
            val_pct, random_validation, label_type, part_options
        )

