from pytorch_lightning import LightningModule
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

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
        self.log('loss_train', loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.train_acc(preds, targets)
        self.log('train_acc')
        return loss
    def validation_step(self, batch, batch_idx):
        scores = self(batch)
        preds = scores.argmax(dim=1)
        targets = batch.labels
        loss = cross_entropy(scores, targets)
        batch_size = len(targets)
        self.log(loss, 'train_loss', batch_size=batch_size, on_step=True, on_epoch=True)
        self.train_acc(preds, targets)