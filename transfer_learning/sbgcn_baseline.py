from pytorch_lightning import LightningModule
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os

from automate import PartFeatures, SBGCN
from automate.sbgcn import LinearBlock

from .datasets import BRepDataModule

def automate_options():
    feature_opts = PartFeatures()

    # No Part Level Features
    feature_opts.face_samples = False
    feature_opts.edge_samples = False
    feature_opts.bounding_box = False
    feature_opts.volume = False
    feature_opts.center_of_gravity = False
    feature_opts.moment_of_inertia = False
    feature_opts.surface_area = False
    feature_opts.samples = False
    feature_opts.mesh = False
    feature_opts.mesh_to_topology = False
    feature_opts.mcfs = False

    # Face Features
    feature_opts.face.parametric_function = True
    feature_opts.face.parameter_values = True
    feature_opts.face.exclude_origin = True

    feature_opts.face.surface_area = True
    feature_opts.face.bounding_box = True
    feature_opts.face.moment_of_inertia = True
    feature_opts.face.center_of_gravity = True

    feature_opts.face.orientation = False
    feature_opts.face.circumference = False
    feature_opts.face.na_bounding_box = False

    # Loop Features
    feature_opts.loop.type = True
    feature_opts.loop.length = True
    feature_opts.loop.center_of_gravity = True
    feature_opts.loop.moment_of_inertia = True
    feature_opts.loop.na_bounding_box = False

    # Edge Features
    feature_opts.edge.parametric_function = True
    feature_opts.edge.orientation = True
    feature_opts.edge.length = True
    feature_opts.edge.bounding_box = True
    feature_opts.edge.center_of_gravity = True
    feature_opts.edge.moment_of_inertia = True
    feature_opts.edge.t_range = False
    feature_opts.edge.start = False
    feature_opts.edge.end = False
    feature_opts.edge.mid_point = False
    feature_opts.edge.na_bounding_box = False

    # Vertex Features
    feature_opts.vertex.position = True

    part_options = {'collect_inferences': False,
    'default_mcfs': False,
    'just_bb': False,
    'normalize': False,
    'num_random_samples': 0,
    'num_sdf_samples': 0,
    'num_uv_samples': 0,
    'sample_normals': False,
    'sample_tangents': False,
    'sdf_sample_quality': 0,
    'tesselate': False,
    'transform': False}

    return part_options, feature_opts, (42, 23, 46, 3, 64, 6)

class SBGCNBaseline(LightningModule):
    def __init__(self, sbgcn_options, out_hidden_size, out_size):
        super().__init__()
        self.sbgcn = SBGCN(*sbgcn_options)
        self.output_mlp = LinearBlock(128, out_hidden_size, out_size, last_linear=True)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
    def forward(self, data):
        _, _, x_f, _, _, _ = self.sbgcn(data)
        return self.output_mlp(x_f)
    def training_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        targets = batch.labels
        loss = cross_entropy(scores, targets)
        batch_size = len(targets)
        self.log('train_loss', loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.train_acc(preds, targets)
        self.log('train_acc', self.train_acc, batch_size, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        targets = batch.labels
        loss = cross_entropy(scores, targets)
        batch_size = len(targets)
        self.log(loss, 'val_loss', batch_size=batch_size, on_step=True, on_epoch=True)
        self.val_acc(preds, targets)
        self.log('val_acc', self.val_acc, batch_size, on_step=False, on_epoch=True)
    def test_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        targets = batch.labels
        batch_size = len(targets)
        self.test_acc(preds, targets)
        self.log('test_acc', self.test_acc, batch_size, on_step=False, on_epoch=True)

def run_experiments(
    index_path, 
    ds_path, 
    experiment_sizes, 
    seeds, 
    cache_dir,
    log_dir,
    val_frac=0.2, 
    batch_size=32,):
    
    ds_name = os.path.basename(index_path).split('.')[0]
    
    part_options, feature_options, sbgcn_options = automate_options()
    

    for seed in seeds:
        for exp_size in experiment_sizes:
            datamodule = BRepDataModule(
                index_path,
                ds_path,
                part_options=part_options,
                implicit=False,
                feature_options=feature_options,
                val_frac=val_frac,
                seed = seed,
                batch_size=batch_size,
                train_size=exp_size,
                cache_dir=cache_dir,
                memcache=True
            )
            exp_name = f'{ds_name}_sbgcn_{exp_size}_{seed}'
            model = SBGCNBaseline(sbgcn_options, 64)
            callbacks = [
                    #EarlyStopping(monitor='val_loss', mode='min', patience=100),
                    ModelCheckpoint(monitor='val_loss', save_top_k=1, filename="{epoch}-{val_loss:.6f}",mode="min"),
                ]
            logger = TensorBoardLogger(log_dir, exp_name)
            trainer = pl.Trainer(max_epochs=-1, logger=logger, callbacks=callbacks, gpus=1)
            trainer.fit(model,datamodule)
            results = trainer.test(datamodule=datamodule)
            test_accuracy = results[0]['test_acc']
            