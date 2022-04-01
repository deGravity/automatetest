from turtle import distance
import pandas as ps
import os
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
import logging
import torch_geometric as tg
import h5py
from automate.transforms import *
from typing import Any, List, Optional, Tuple, Union
import math


class SavedDataModule(pl.LightningDataModule):

    # @staticmethod
    # def add_module_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("SavedDataModule")
    #     parser.add_argument("--torch_path", type=str)
    #     parser.add_argument("--debug_dataset", type=str)
    #     parser.add_argument("--pair_data_path", type=str)
    #     parser.add_argument("--filters", type=str, nargs='+')
    #     parser.add_argument("--exclusions", type=str, nargs='+')
    #     parser.add_argument("--splits", type=str, nargs='+')
    #     parser.add_argument("--batch_size", type=int, default=int)
    #     parser.add_argument("--shuffle", type=bool, default=True)
    #     parser.add_argument("--num_workers", type=int, default=10)
    #     parser.add_argument("--persistent_workers", type=bool, default=True)
    #     return parent_parser

    # @classmethod
    # def from_argparse_args(cls, args, **kwargs):
    #     return pl.utilities.argparse.from_argparse_args(cls, args, **kwargs)

    # @classmethod
    # def add_argparse_args(cls, parent_parser):
    #     parser = pl.utilities.argparse.add_argparse_args(cls, parent_parser)
    #     return parser

    def __init__(
        self,
        torch_path: str,
        pair_data_path: str,
        splits_path: str,
        filters: List[str] = [],
        exclusions: List[str] = [],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 10,
        persistent_workers: bool = True,
        debug_dataset: bool = False,
        corrective_transforms: bool = True,
        motion_pointnet: bool = False,
        num_points: int = 100,
        motion_displacement: float = 0.05,
        motion_angle: float = math.pi / 16,
        pair_pointnet: bool = False,
        assembly_points: bool = False
    ):
        super().__init__()
        self.path = torch_path
        self.pair_data_path = pair_data_path
        self.filters = filters
        self.exclusions = exclusions
        self.splits = [os.path.join(splits_path, 'train.txt'), os.path.join(splits_path, 'test.txt'), os.path.join(splits_path, 'validation.txt')]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and (self.num_workers > 0)
        self.debug = debug_dataset
        self.transforms = []
        if corrective_transforms:
            self.transforms += [fix_edge_sets, remap_type_labels]
        if motion_pointnet:
            self.transforms += [sample_motions(num_points, motion_displacement, motion_angle)]
        if pair_pointnet:
            self.transforms += [sample_points(num_points, assembly_points=assembly_points)]
    
    def setup(self, **kwargs):
        #todo: transforms
        logging.info('train dataset')
        self.train = SavedDataset(self.splits[0], prefix_path=self.path, filter_file=self.filters, exclusion_file=self.exclusions, debug=self.debug, pair_data_path=self.pair_data_path, transforms=self.transforms)
        logging.info('test dataset')
        self.test = SavedDataset(self.splits[1], prefix_path=self.path, filter_file=self.filters, exclusion_file=self.exclusions, debug=self.debug, pair_data_path=self.pair_data_path, transforms=self.transforms)
        logging.info('validation dataset')
        self.validate = SavedDataset(self.splits[2], prefix_path=self.path, filter_file=self.filters, exclusion_file=self.exclusions, debug=self.debug, pair_data_path=self.pair_data_path, transforms=self.transforms)
        

    def train_dataloader(self):
        return tg.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=self.shuffle, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return tg.data.DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return tg.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False, persistent_workers=self.persistent_workers)
    


def filter_data(index, filter_file, exclude=False):
    if not isinstance(filter_file, list):
        filter_file = [filter_file]
    for filter in filter_file:
        if filter.endswith('.csv'):
            df = ps.read_csv(filter, header=None)
            filter_set = set(df[0])
            
        elif filter.endswith('.txt'):
            with open(filter) as f:
                filter_set = {int(l) for l in f.readlines()}
        elif filter.endswith('.parquet'):
            df = ps.read_parquet(filter)
            filter_set = set(df.index)
        if exclude:
            index = [ind for ind in index if ind not in filter_set]
        else:
            index = [ind for ind in index if ind in filter_set]
    return index


class SavedDataset(Dataset):
    def __init__(self, index_file,
                pair_data_path,
                prefix_path=None,
                filter_file=None,
                exclusion_file=None,
                transforms=[],
                distance_threshold=0.01,
                debug=False):
        
        self.pair_data_path = pair_data_path
        self.debug = debug
        self.distance_threshold = distance_threshold
        self.transform = compose(*transforms[::-1]) if transforms else None

        with open(index_file) as f:
            self.index = [int(l.rstrip()) for l in f.readlines()]
        
        if filter_file is not None:
            self.index = filter_data(self.index, filter_file)

        if exclusion_file is not None:
            self.index = filter_data(self.index, exclusion_file, exclude=True)
        
        self.ids = self.index
        if prefix_path is not None:
            self.index = [os.path.join(prefix_path, str(l)) for l in self.index]
        
        self.index = [str(path) + '.dat' for path in self.index]
        
        logging.info(f'dataset size: {len(self.index)} elements')
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i):
        path = self.index[i]
        data = torch.load(path)
        data.ass = int(os.path.split(path)[1].split('.')[0])
        
        close_pairs = []
        type_labels = []
        axis_labels = []

        mc_path = os.path.join(self.pair_data_path, f'{self.ids[i]}.hdf5')
        with h5py.File(mc_path, 'r') as f:
            pair_data = f['pair_data']
            for key in pair_data.keys():
                if pair_data[key].attrs['distance'] < self.distance_threshold and len(pair_data[key]['axes'].keys()) > 0:
                    pair = tuple(int(k) for k in key.split(','))
                    close_pairs.append(pair)
                    typeLabel = pair_data[key].attrs['type']
                    originalLabel = typeLabel
                    axisIndices = []
                    axisDirIndices = []
                    axisIndex = pair_data[key].attrs['axisIndex']
                    dirIndex = pair_data[key].attrs['dirIndex']
                    if typeLabel < 0:
                        typeLabel = pair_data[key].attrs['augmented_type']
                        axisIndex = pair_data[key].attrs['augmented_axisIndex']
                        dirIndex = pair_data[key].attrs['augmented_dirIndex']
                    if axisIndex < 0 and typeLabel == 7:
                        dirkey = list(pair_data[key]['axes'].keys())[0]
                        axisIndex = list(pair_data[key]['axes'][dirkey]['indices'])[0]
                    if axisIndex < 0 and typeLabel == 3:
                        axisIndex = list(pair_data[key]['axes'][str(dirIndex)]['indices'])[0]
                    if self.debug:
                        assert(typeLabel >= 0)
                        assert(axisIndex >= 0)
                    type_labels.append(typeLabel)
                    axis_labels.append(axisIndex)
                    all_dir_indices = list(pair_data[key]['dirs']['indices'])
        
        data.part_edges = torch.tensor(close_pairs).T
        data.__edge_sets__['part_edges'] = ['graph_idx', 'graph_idx']
        data.mate_labels = torch.tensor(type_labels)
        data.axis_labels = torch.tensor(axis_labels)
        

        if self.transform is not None:
            data = self.transform(data)

        return data


if __name__ == '__main__':
    transforms = [fix_edge_sets, remap_type_labels, sample_motions(100, .05, math.pi/16)]
    dataset = SavedDataset('/fast/jamesn8/assembly_data/assembly_torch2_fixsize/full_pipeline/train.txt', '/fast/jamesn8/assembly_data/assembly_torch2_fixsize/full_pipeline/axis_data', prefix_path='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/full_pipeline/batches/', exclusion_file=['/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/assemblies_with_discrepant_mcs.txt', '/fast/jamesn8/assembly_data/assembly_torch2_fixsize/test_add_mate_labels/stats.parquet'], transforms = transforms, debug=True)
    for i in range(100):
        print(i,dataset.index[i])
        data = dataset[i]
        print(data)