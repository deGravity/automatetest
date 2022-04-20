from automate.implicit import preprocess_file,preprocess_implicit_part
from pspy import ImplicitPart
from argparse import ArgumentParser
from zipfile import ZipFile
import os
import json
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source')
    parser.add_argument('--dest')
    parser.add_argument('--ext')
    parser.add_argumest('--list')

    args = parser.parse_args()

    print(f'{args.source=}')
    print(f'{args.dest=}')
    print(f'{args.ext=}')

    # Todo - make arguments
    samples = 500
    ref_samples = 5000
    normalize = True

    # TODO - add multiprocessing + num-threads argument
    
    if args.source.endswith('.zip'):
        with ZipFile(args.source, 'r') as zf:
            to_process = [name for name in zf.namelist() if name.endswith(args.ext)]
        for file in tqdm(to_process):
            save_path = os.path.join(args.dest, f'{file}.pt')
            if os.path.exists(save_path):
                continue
            with zf.open(file, 'r') as f:
                file_text = f.read().decode('utf-8')
            ipart = ImplicitPart(file_text, samples, ref_samples, normalize) 
            preprocess_implicit_part(ipart, save_path, samples)
        exit(0)
    
    with open(args.list, 'r') as f:
        file_list = json.load(f)
    
    # Flatten file list if dict
    if isinstance(file_list, dict):
        all_files = []
        for l in file_list.values():
            all_files += list(file_list)
        to_process = all_files
    else:
        to_process = file_list

    for id in tqdm(to_process):
        preprocess_file(args.source, id, args.ext, args.dest, samples, ref_samples, normalize)

