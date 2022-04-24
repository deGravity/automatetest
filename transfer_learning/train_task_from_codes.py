from train_latent_space import BRepFaceEncoder, BRepFaceAutoencoder, BRepDS
import json
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

class CodeDS(torch.utils.data.Dataset):
    def __init__(self, path, mode, validate_pct = 0):
        self.data = torch.load(path)

if __name__ == '__main__':
    pass