import torch
import torch_geometric as tg
import pytorch_lightning as pl
from torch_scatter import scatter_mean
from . import PartDataset, SBGCN, LinearBlock


class UVPartDataset(PartDataset):
    def __init__(
        self,
        splits_path = '/projects/grail/benjonesnb/cadlab/data-release/splits.json',
        data_dir = '/fast/benjones/data/',
        mode = 'train',
        cache_dir = None,
        num_samples = 10
    ):
        super().__init__(splits_path, data_dir, mode, cache_dir, None, None)
       
        self.num_samples = num_samples
        self.options.num_uv_samples = self.num_samples
        self.options.normalize = False#True
        self.options.tesselate = False
        self.options.collect_inferences = False
        self.options.default_mcfs = False
        self.options=None # Hack for GPUs since this won't pickle - only works if the cache is already built!

        # Turn off all non-parametric function features
        self.features.face.center_of_gravity = False
        self.features.face.bounding_box = False
        self.features.face.center_of_gravity = False
        self.features.face.circumference = False
        self.features.face.exclude_origin = False
        self.features.face.na_bounding_box = False
        self.features.face.surface_area = False
        self.features.face.moment_of_inertia = False

        self.features.edge.bounding_box = False
        self.features.edge.center_of_gravity = False
        self.features.edge.end = False
        self.features.edge.exclude_origin = False
        self.features.edge.length = False
        self.features.edge.mid_point = False
        self.features.edge.moment_of_inertia = False
        self.features.edge.na_bounding_box = False
        self.features.edge.start = False
        self.features.edge.t_range = False

        self.features.loop.center_of_gravity = False
        self.features.loop.length = False
        self.features.loop.moment_of_inertia = False
        self.features.loop.na_bounding_box = False

        # Turn off part-level features
        self.features.mcfs = False
        self.features.volume = False
        self.features.surface_area = False
        self.features.center_of_gravity = False
        self.features.bounding_box = False
        self.features.moment_of_inertia = False
        
        # Turn off mesh features
        self.features.mesh = True
        self.features.mesh_to_topology = True

class UVFitter(pl.LightningModule):
    def __init__(
        self,
        sbgcn_size = 64,
        linear_sizes = (512, 512),
        n_samples = 10,
        f_in = 22,
        l_in = 10,
        e_in = 19,
        v_in = 3
    ):
        super().__init__()
        grid_size = 9*n_samples*n_samples
        self.sbgcn = SBGCN(f_in, l_in, e_in, v_in, sbgcn_size, 0)
        self.lin = LinearBlock(sbgcn_size, *linear_sizes, grid_size, last_linear=True)
        self.loss = torch.nn.MSELoss()
    def forward(self, graph):
        _, _, x_f, _, _, _ = self.sbgcn(graph)
        preds = self.lin(x_f)
        return preds
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self, data, batch_idx):
        target = data.face_samples.reshape((data.face_samples.size(0),-1))
        preds = self(data)
        error = self.loss(preds, target)
        self.log('train_loss', error,batch_size=32)
        return error
    def validation_step(self, data, batch_idx):
        target = data.face_samples.reshape((data.face_samples.size(0),-1))
        preds = self(data)
        error = self.loss(preds, target)
        self.log('val_loss', error,batch_size=32)

        #face_classes = data.faces[:,:13].argmax(dim=1).max()
        #face_errors = torch.nn.functional.mse_loss(preds, target, reduction='none')
        #class_errors = scatter_mean(face_errors, face_classes)
        #for i in range(len(class_errors)):
        #    self.log(f'val_loss_class_{i}', class_errors[i])
    def test_step(self, data, batch_idx):
        target = data.face_samples.reshape((data.face_samples.size(0),-1))
        preds = self(data)
        error = self.loss(preds, target)
        self.log('test_loss', error,batch_size=32)


class UVPartDataModule(pl.LightningDataModule):
    def __init__(
        self,
        splits_path = '/projects/grail/benjonesnb/cadlab/data-release/splits.json',
        data_dir = '/fast/benjones/data/',
        cache_dir = None,
        num_samples = 10,
        batch_size = 32,
        num_workers = 10,
        shuffle=True
    ):
        super().__init__()
        self.splits_path = splits_path
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, **kwargs):
        self.train = UVPartDataset(self.splits_path, self.data_dir, 'train', self.cache_dir, self.num_samples)
        self.test = UVPartDataset(self.splits_path, self.data_dir, 'test', self.cache_dir, self.num_samples)
        self.val = UVPartDataset(self.splits_path, self.data_dir, 'validate', self.cache_dir, self.num_samples)

    def train_dataloader(self):
        return tg.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=self.shuffle)

    def val_dataloader(self):
        return tg.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)

    def test_dataloader(self):
        return tg.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)

