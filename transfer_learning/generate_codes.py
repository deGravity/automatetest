from train_latent_space import BRepFaceEncoder, BRepFaceAutoencoder, BRepDS
import json
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

if __name__ == '__main__':
    ds = BRepDS(
        'D:/fusion360segmentation/simple_train_test.json', 
        'D:/fusion360segmentation/simple_preprocessed', 
        'test',
        preshuffle=False,
        validate_pct=0,
        label_root='D:/fusion360segmentation/seg',
        label_ext='seg'
    )
    dl = DataLoader(ds, batch_size=1)

    checkpoint_path = r'D:\fusion360segmentation\runs\encdec\nofilter64e3\version_0\checkpoints\epoch=47-val_loss=0.001085.ckpt'

    model = BRepFaceAutoencoder(64, 1024,4)
    sd = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(sd)

    codes = [model.encode(batch) for batch in tqdm(dl)]

    code_output_path = r'D:\fusion360segmentation\codes64_test.pt'

    torch.save(codes, code_output_path)




