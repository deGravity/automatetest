import torch
import torch_geometric as tg
import pytorch_lightning as pl
from torch_scatter import scatter_mean
from . import PartDataset, SBGCN, LinearBlock, part_to_graph
from pspy import PartOptions, Part
import os

class SimplePartDataset(torch.utils.data.Dataset):
    def __init__(self, path, n_samples = 10, normalize = True, mode='train'):
        super().__init__()
        self.path = path
        self.n_samples = n_samples
        self.normalize = normalize
        self.mode = mode
        self.files = [os.path.join(path,x) for x in os.listdir(path) if x.endswith('.pt')]
        cutoff = int(len(self.files) * 0.95)
        train_files = self.files[:cutoff]
        val_files = self.files[cutoff:]
        self.files = {
            'train':train_files,
            'validate':val_files
        }
    
    def __len__(self):
        return len(self.files[self.mode])
    def __getitem__(self, idx):
        data = torch.load(self.files[self.mode][idx])

        # Normalization
        if self.normalize:
            face_points = data.face_samples[:,:3,:,:].permute((0,2,3,1)).reshape((-1,3))
            edge_points = data.edge_samples[:,:3,:].permute(0,2,1).reshape((-1,3))
            points = torch.cat([face_points, edge_points],dim=0)
            min_point = points.min(dim=0).values
            max_point = points.max(dim=0).values
            center_point = (min_point + max_point) / 2
            scale = ((max_point - min_point) / 2).max()

        face_types = data.faces[:,:5]#[:,:13]
        face_oris = data.faces[:,-1].reshape((-1,1))
        face_params = data.faces[:,13:-1]
        if self.normalize:
            face_params[:,:3] = (face_params[:,:3] - center_point) / scale
            face_params[:,9] = face_params[:,9] / scale
            toruses = (face_types.argmax(dim=1) == 4)
            face_params[toruses,10] = face_params[toruses,10] / scale # only torus scales last param
        data.faces = torch.cat([face_types, face_params, face_oris],dim=1)

        edge_types =  data.edges[:,:3]#[:,:11]
        edge_params = data.edges[:,11:-1]
        if self.normalize:
            edge_params[:,:3] = (edge_params[:,:3] - center_point) / scale
            edge_params[:,9:] = (edge_params[:,9:] / scale)
        edge_oris = data.edges[:,-1].reshape((-1,1))
        data.edges = torch.cat([edge_types, edge_params, edge_oris],dim=1)

        face_xyz = data.face_samples[:,:3,:,:].permute(0,2,3,1)
        if self.normalize:
            face_xyz = (face_xyz - center_point) / scale
        face_mask = data.face_samples[:,-1:,:,:].permute(0,2,3,1)
        data.face_samples = torch.cat([face_xyz, face_mask],dim=3)

        edge_xyz = data.edge_samples[:,:3,:].permute(0,2,1)
        if self.normalize:
            edge_xyz = (edge_xyz - center_point) / scale
        data.edge_samples = edge_xyz


        #d_samps = data.face_samples.shape[-1]
        #stride = d_samps // self.n_samples

        #data.face_samples = data.face_samples.permute((0,2,3,1))[:,::stride,::stride,:]
        #data.edge_samples = data.edge_samples.permute((0,2,1))[:,::stride,:]

        return data

class UVPartDataset(PartDataset):
    def __init__(
        self,
        splits_path = '/projects/grail/benjonesnb/cadlab/data-release/splits.json',
        data_dir = '/fast/benjones/data/',
        mode = 'train',
        cache_dir = None,
        num_samples = 10,
        get_mesh = False,
        cache_only = False
    ):
        super().__init__(splits_path, data_dir, mode, cache_dir, None, None)
       
        self.options = None

        self.num_samples = num_samples
        self.get_mesh = get_mesh
        
        if cache_only:
            self.options=None # Hack for GPUs since this won't pickle - only works if the cache is already built!

        # Turn off all non-parametric function features
        self.features.face.parametric_function = True
        self.features.face.parameter_values = True
        self.features.face.orientation = True

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
        self.features.edge.orientation = True
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
        
        # Turn on/off mesh features
        self.features.mesh = self.get_mesh
        self.features.mesh_to_topology = self.get_mesh
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(len(self.part_paths))[idx]]
        if not self.cache_dir is None:
            cache_file = os.path.join(self.cache_dir, self.mode, f'{idx}.pt')
            if os.path.exists(cache_file):
                return torch.load(cache_file)
        part_path = os.path.join(self.data_dir, self.part_paths[idx])
        if part_path.endswith('.pt'):
            part = torch.load(part_path)
        else:
            options = PartOptions()
            options.num_uv_samples = self.num_samples
            options.normalize = False
            options.tesselate = self.get_mesh
            options.collect_inferences = False
            options.default_mcfs = False
            part = Part(part_path, options)

        graph = part_to_graph(part, self.features)
        if not self.cache_dir is None:
            torch.save(graph, cache_file)
        return graph

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
        get_mesh = False,
        cache_only = False,
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
        self.get_mesh = get_mesh
        self.cache_only = cache_only

    def setup(self, **kwargs):
        self.train = UVPartDataset(self.splits_path, self.data_dir, 'train', self.cache_dir, self.num_samples, self.get_mesh, self.cache_only)
        self.test = UVPartDataset(self.splits_path, self.data_dir, 'test', self.cache_dir, self.num_samples, self.get_mesh, self.cache_only)
        self.val = UVPartDataset(self.splits_path, self.data_dir, 'validate', self.cache_dir, self.num_samples, self.get_mesh, self.cache_only)

    def train_dataloader(self):
        return tg.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=self.shuffle)

    def val_dataloader(self):
        return tg.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)

    def test_dataloader(self):
        return tg.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False)

