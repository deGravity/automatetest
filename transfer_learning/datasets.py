from collections import Counter
import json
from math import ceil
import os
import random
from zipfile import ZipFile

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from automate import part_to_graph, PartFeatures, implicit_part_to_data
from pspy import Part, ImplicitPart, PartOptions

def create_subset(data, seed, size):
    random.seed(seed)
    return random.sample(data, size)

class DataSubset(Dataset):
    def __init__(self, ds, size, seed, mode='train', val_frac=0.0, no_stratify=False):
        super().__init__()
        self.ds = ds
        if size > 0 and size < 1:
            fraction = size # Used for stratified sampling
            size = ceil(len(ds)*size)
        elif size < 0:
            size = len(ds)
        
        self.size = size
        self.seed = seed

        stratify_sampling = False
        if not no_stratify and hasattr(ds, 'labels'):
            if all([len(l) == 1 for l in ds.labels]):
                stratify_sampling = True

        if stratify_sampling:
            category_counts = Counter([l[0] for l in ds.labels])
            num_categories = max([l[0] for l in ds.labels])
            cat_seed = lambda cat_enc: cat_enc + seed*num_categories
            def subset_category(cat, fraction):
                return create_subset(
                    [i for i,l in enumerate(ds['labels']) if l[0] == cat],
                    cat_seed(cat),
                    max(2, ceil(fraction*category_counts[cat]))
                )
            indices = [i for cat in category_counts for i in subset_category(cat, fraction)]
            val_frac = val_frac if ceil(val_frac*len(indices) >= num_categories) else num_categories
        else:
            indices = create_subset(range(len(ds)), seed, size)

        if mode in ['train', 'validate'] and val_frac > 0:
            train_indices, val_indices = train_test_split(
               indices, test_size=val_frac, random_state=seed,
               stratify=[ds['labels'][i][0] for i in indices] if stratify_sampling else None
            )
            indices = train_indices if mode == 'train' else val_indices
        
        self.indices = indices

        
    def __getitem__(self, idx):
        return self.ds[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class CachedDataset(Dataset):
    def __init__(self, ds, cache_dir = None, memcache = False):
        super().__init__()
        self.ds = ds
        self.memcache = memcache
        self.cache_dir = cache_dir
        if hasattr(ds, 'labels'):
            self.labels = ds.labels
        self.cache = None

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
    
    def cache_file(self, val, idx):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f'{idx}.pt')
            if not os.path.exists(cache_path):
                cache_dir = os.path.dirname(cache_path)
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(val, cache_path)
    
    def cache_mem(self, val, idx):
        if self.memcache:
            if not self.cache:
                self.cache = dict()
            self.cache[idx] = val
        

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
        self.datafile = None

    def get_options_struct(self):
        options = PartOptions()
        for k,v in self.part_options.items():
            if hasattr(options, k):
                setattr(options, k, v)
        return options

    def __len__(self):
        return len(self.ids)

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

class BRepDataModule(LightningDataModule):
    def __init__(
        self,
        index_path: str, 
        data_path: str,  
        part_options: dict = {},
        implicit: bool = True,
        feature_options: PartFeatures = PartFeatures(),
        implicit_options: dict = {},
        val_frac = 0.2,
        seed=0,
        batch_size=32,
        train_size=-1,
        cache_dir=None,
        memcache=True,
        no_stratify=False
    ):
        super().__init__()
        part_ds_train = PartDataset(
            index_path,
            data_path,
            mode = 'train',
            part_options=part_options,
            implicit = implicit,
            feature_options=feature_options,
            implicit_options=implicit_options
        )
        part_ds_test = PartDataset(
            index_path,
            data_path,
            mode = 'test',
            part_options=part_options,
            implicit = implicit,
            feature_options=feature_options,
            implicit_options=implicit_options
        )

        cache_ds_train = CachedDataset(part_ds_train, memcache=memcache, cache_dir=os.path.join(cache_dir,'train'))
        self.ds_train = DataSubset(cache_ds_train, train_size, seed, val_frac=val_frac, mode='train',no_stratify=no_stratify)
        self.ds_val = DataSubset(cache_ds_train, train_size, seed, val_frac=val_frac, mode='validate', no_stratify=no_stratify)
        self.ds_test = CachedDataset(part_ds_test, memcache=memcache, cache_dir=os.path.join(cache_dir,'test'))

        self.batch_size = min(batch_size, len(self.ds_train))



        
    
