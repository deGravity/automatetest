import numpy as np
import altair as alt
import os

import json
from matplotlib import pyplot as plt
from zipfile import ZipFile
from tqdm import tqdm
from pspy import Part
from rendering import render_part2, render_grid
import pandas as pd
from util import load_json

def plot_accuracies(frame, dataset, scale='log'):
    line = alt.Chart(frame).mark_line().encode(
        x=alt.X('train_size',scale=alt.Scale(type=scale)),
        y=alt.Y('mean(accuracy)'),
        color=alt.Color('model',sort=['Ours','UV-Net','BRepNet'])
    )

    band = alt.Chart(frame).mark_errorband(extent='ci').encode(
        x=alt.X('train_size', scale=alt.Scale(type=scale)),
        y=alt.Y('accuracy', title='accuracy'),
        color=alt.Color('model',sort=['Ours','UV-Net','BRepNet'])
    )

    return (band + line).properties(
        title=f'{dataset} Accuracy vs Train Size'
    )

def plot_segmentation_accuracies(frame, dataset, average='micro', scale='log'):
    frame = frame[frame.dataset == dataset]
    if average=='macro':
        frame = frame.groupby(
            ['dataset','model','train_size','seed','test_idx']
            ).agg({'accuracy':'mean'}).reset_index()
    frame = frame.groupby(
        ['dataset','model','train_size','seed']).agg({'accuracy':'mean'}).reset_index()
    return plot_accuracies(frame, dataset, scale)

def plot_classification_accuracies(frame, dataset, size_proto_model='Ours', scale='log'):
    ts_frac = frame[frame.model==size_proto_model].groupby('train_fraction').agg({'train_size':'min'}).reset_index()
    tf_to_ts = dict(zip(ts_frac.train_fraction,ts_frac.train_size))
    frame.train_size = np.vectorize(lambda x: tf_to_ts[x])(frame.train_fraction)
    frame = frame.groupby(['dataset','model','train_size','seed']).agg({'accuracy':'mean'}).reset_index()
    return plot_accuracies(frame, dataset, scale)

def plot_reconstruction_vs_classification(accs):
    accs['one'] = 1
    iscorr = np.vectorize(lambda x: 'correct' if x == 1 else 'incorrect')
    accs['correct'] = iscorr(accs.label)
    accs['logmse'] = np.log(accs.mse)


    alt.data_transformers.disable_max_rows()

    return alt.Chart(accs[accs.mse < 1]).mark_bar().encode(
        x=alt.X('logmse',bin=alt.Bin(maxbins=50)),
        y=alt.Y('sum(one)', stack='normalize', axis=alt.Axis(format='%')),
        color=alt.Color('label:N')
    ).facet(row='train_size')

def render_segmentation_comparisons(
    root = '../../repbrep/', 
    dataset_name = 'fusion360seg', 
    camera_name = 'f360seg', 
    dataset = 'Fusion360Seg',
    seed = 0,
    train_size = 100,
    num_to_render = 10, 
    render_size = 5,
    max_labels = 8
):


    repbrep = root
    fusion360seg_zip_path = os.path.join(repbrep, 'datasets', f'{dataset_name}.zip')
    index_path = os.path.join(repbrep, 'datasets', f'{dataset_name}.json')
    poses_path = os.path.join(repbrep, 'datasets', f'{camera_name}_test_poses.npy')
    zooms_path = os.path.join(repbrep, 'datasets', f'{camera_name}_test_zooms.npy')

    print('Loading Segmentation Predictions')
    segmentation_predictions = pd.read_parquet(os.path.join(repbrep, 'results', 'segmentation_predictions.parquet'))

    print('Computing Part Level Accuracies')
    # Compute Part-Level Accuracy at 100 samples from seed 0 for each model
    part_accs = segmentation_predictions.groupby(
        ['dataset','model','train_size','seed','test_idx']).agg(
            {'accuracy':'mean'}).reset_index()
    part_accs = part_accs[(part_accs.seed == seed) & (part_accs.train_size == train_size) & (part_accs.dataset == dataset)].sort_values('test_idx')
    part_accs_ours = part_accs[part_accs.model == 'Ours'].accuracy.values
    part_accs_uv = part_accs[part_accs.model == 'UV-Net'].accuracy.values
    part_accs_brep = part_accs[part_accs.model == 'BRepNet'].accuracy.values

    print('Computing Nuanced Wins')
    # Compute the total accuracy lift over baselines for each test example
    lift = (part_accs_ours - part_accs_brep) + (part_accs_ours - part_accs_uv)

    # Compute the number of faces in each test example as a complexity metric
    complexity = segmentation_predictions[segmentation_predictions.dataset == dataset].groupby(
        'test_idx').agg({'face_idx':'max'}).reset_index().sort_values('test_idx').face_idx.values + 1

    # Sort by lift and other keys, and filter to parts with more than 25 faces, and baselines with at least 10% accuracy
    nuanced_wins = [x[0] for x in sorted(enumerate(zip(lift,complexity, part_accs_ours, part_accs_uv, part_accs_brep)), reverse=True, key = lambda x : x[1]) 
    if x[1][1] > 25 and x[1][3] > .1 and x[1][4] > .1]


    print('Loading Camera Angles')
    all_poses = np.load(poses_path)
    all_zooms = np.load(zooms_path)

    print('Selecting Parts to Render')
    with open(index_path,'r') as f:
        index = json.load(f)
    test_indices_to_render = nuanced_wins[:num_to_render]
    paths_to_render = [index['template'].format(*index['test'][i]) for i in test_indices_to_render]
    poses_to_render = all_poses[test_indices_to_render]
    zooms_to_render = all_zooms[test_indices_to_render]



    gt_labels = []
    our_preds =[]
    uv_preds = []
    brep_preds = []
    seg_preds = segmentation_predictions[
            (segmentation_predictions.dataset == dataset) &
            (segmentation_predictions.seed == 0) &
            (segmentation_predictions.train_size == 100)
        ]
    for i in tqdm(test_indices_to_render, 'Gathering Predictions and Labels'):
        gt_labels.append(seg_preds[(seg_preds.test_idx == i) & (seg_preds.model == 'Ours')].sort_values('face_idx').label.values)
        our_preds.append(seg_preds[(seg_preds.test_idx == i) & (seg_preds.model == 'Ours')].sort_values('face_idx').prediction.values)
        uv_preds.append(seg_preds[(seg_preds.test_idx == i) & (seg_preds.model == 'UV-Net')].sort_values('face_idx').prediction.values)
        brep_preds.append(seg_preds[(seg_preds.test_idx == i) & (seg_preds.model == 'BRepNet')].sort_values('face_idx').prediction.values)

    to_render = list(zip(test_indices_to_render,paths_to_render,poses_to_render,zooms_to_render,gt_labels,our_preds,uv_preds,brep_preds))

    fig, axes = plt.subplots(
        num_to_render, 4, 
        figsize=(4*render_size, num_to_render*render_size),
        gridspec_kw={'wspace':0,'hspace':0},dpi=300
    )

    print('Loading Part Zipfile')
    with ZipFile(fusion360seg_zip_path, 'r') as zf:
        for k, (test_idx, path, pose, zoom, gt, our_pred, uv_pred, brep_pred) in enumerate(tqdm(to_render, 'Rendering Parts')):
            part = Part(zf.open(path).read().decode('utf-8'))
            
            gt_im = render_part2(part, pose, zoom, face_labels=gt, max_labels=max_labels)
            our_im = render_part2(part, pose, zoom, face_labels=our_pred, max_labels=max_labels)
            uv_im = render_part2(part, pose, zoom, face_labels=uv_pred, max_labels=max_labels)
            brep_im = render_part2(part, pose, zoom, face_labels=brep_pred, max_labels=max_labels)

            axes[k,0].imshow(gt_im)
            axes[k,1].imshow(our_im)
            axes[k,2].imshow(uv_im)
            axes[k,3].imshow(brep_im)
            for ax in axes[k]:
                ax.axis('off')


def plot_segmentation_results(seg_preds_path, index_path, gridspec, w=800, h=800):
    seg_preds = pd.read_parquet(seg_preds_path)
    index = load_json(index_path)
    camera_params = np.load(index_path[:-3]+'.npz')
    args_grid = []
    with ZipFile(index_path[:-3]+'.zip') as zf:
        for specrow in gridspec:
            argsrow = []
            for spec in specrow:
                (dataset, model, test_idx, train_size, seed, col, max_labels) = spec
                labels = seg_preds[
                    (seg_preds.dataset == dataset) & 
                    (seg_preds.model == model) & 
                    (seg_preds.test_idx == test_idx) &
                    (seg_preds.train_size == train_size) &
                    (seg_preds.seed == seed)  
                    ].sort_values('face_idx')[col].values
                pose = camera_params['test_poses'][test_idx]
                zoom = camera_params['test_zooms'][test_idx]
                part_name = index['template'].format(*index['test'][test_idx])
                part = Part(zf.open(part_name, 'r').read().decode('utf-8'))
                render_args = (part, pose, zoom, max_labels, labels)
                argsrow.append(render_args)
        args_grid.append(argsrow)
    return render_grid(args_grid, w, h)


