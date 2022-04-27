import argparse
from multiprocessing import Pool, Process
from pspy import ImplicitPart
from argparse import ArgumentParser
from automate import implicit_part_to_data
import os
import torch
import json
from tqdm import tqdm

def preprocess(filepath, outpath):
    try:
        ipart = ImplicitPart(filepath, 500, 5000, True)
        if ipart.valid:
            data = implicit_part_to_data(ipart, 500)
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            torch.save(data, outpath)
    except Exception as e:
        pass
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--filelist')
    parser.add_argument('--dataroot')

    args = parser.parse_args()

    pool = Pool(processes=args.processes)

    with open(args.filelist, 'r') as f:
        files = json.load(f)
    
    if isinstance(files, dict):
        all_files = []
        for l in files.values():
            all_files = all_files + l
    else:
        all_files = files
    
    for filepath in tqdm(all_files):
        outpath = os.path.join(args.dataroot, f'{filepath}.pt')
        pool.a
