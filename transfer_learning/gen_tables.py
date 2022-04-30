import numpy as np
from pylatex import Tabular, MultiColumn
from pylatex.utils import NoEscape
from argparse import ArgumentParser
import os
import json

def format_experiments(
    experiments,
    preferred_metrics = dict(),
    metric_renames = dict(),
    size_renames = dict(),
    model_order = [], # [str]
    dataset_order = [], # [str]
    omitted_experiments = [], # [(ds, model)]
    omitted_datasets = [], # [str]
    ommitted_models = [], #  [str]
    placeholder_experiments = [] # [(ds, model, metric, num_sizes),...]
):

    metric_transform = lambda x: metric_renames.get(x,x) if isinstance(metric_renames, dict) else metric_renames
    size_transform = lambda x : size_renames.get(x,x) if isinstance(size_renames, dict) else size_renames

    omitted_experiments = [tuple(x) for x in omitted_experiments]

    summarized_experiments = {}
    for exp in experiments:
        exp_dataset = exp['dataset']
        if exp_dataset in omitted_datasets:
            continue
        exp_model = exp['model']
        if exp_model in ommitted_models:
            continue
        if (exp_dataset, exp_model) in omitted_experiments:
            continue
        exp_metrics = {}
        for k,v in exp.items():
            if isinstance(v, dict):
                metric = {'name':k,'size':len(v),'sizes':[],'means':[],'stds':[]}
                for s,m in v.items():
                    mean = np.mean(m)
                    std = np.std(m)
                    metric['sizes'].append(s)
                    metric['means'].append(mean)
                    metric['stds'].append(std)
                exp_metrics[k] = metric
        experiment = {
            'dataset':exp_dataset,
            'model': exp_model,
            'metrics': exp_metrics
        }
        if exp_dataset not in summarized_experiments:
            summarized_experiments[exp_dataset] = []
        summarized_experiments[exp_dataset].append(experiment)

    # Add in placeholder experiments

    for ds,model,metric,size in placeholder_experiments:
        if ds not in summarized_experiments:
            summarized_experiments[ds] = []
        existing_models = [se['model'] for se in summarized_experiments[ds]]
        if model in existing_models:
            continue
        summarized_experiments[ds].append({
            'dataset':ds,
            'model':model,
            'metrics':{metric:{
                'name':metric,
                'size':size
            }}
        })

    # Find/Assert Compatible Metrics and Choose per Dataset

    dataset_metrics = {}
    for dataset in summarized_experiments:
        d_exps = summarized_experiments[dataset]
        all_metrics = [set(exp['metrics'].keys()) for exp in d_exps]
        potential_metrics = set.intersection(*all_metrics)
        assert(len(potential_metrics) > 0)
        consistent_metrics = []
        for metric in potential_metrics:
            sizes = {exp['metrics'][metric]['size'] for exp in d_exps}
            if len(sizes) == 1:
                consistent_metrics.append(metric)
        assert(len(consistent_metrics) > 0)
        if preferred_metrics.get(dataset,'') in consistent_metrics:
            chosen_metric = preferred_metrics[dataset]
        else:
            chosen_metric = consistent_metrics[0]
        dataset_metrics[dataset] = chosen_metric

    def modelrank(m):
        return model_order.index(m) if m in model_order else len(model_order)
    def dsrank(ds):
        return dataset_order.index(ds) if ds in dataset_order else len(dataset_order)

    ds_order = sorted(list(dataset_metrics.keys()),key=dsrank)

    max_num_sizes = 0
    result_groups = []
    for ds in ds_order:
        ds_experiments = summarized_experiments[ds]
        metric = dataset_metrics[ds]
        train_sizes = None
        num_sizes = None
        for exp in ds_experiments:
            met = exp['metrics'][metric]
            num_sizes = met['size']
            if 'sizes' in met:
                train_sizes = met['sizes']
                break
        if train_sizes is None:
            train_sizes_str = ['--'] * num_sizes
            train_sizes = [0] * num_sizes
        else:
            train_sizes_str = [str(size_transform(x)) for x in train_sizes]
        max_num_sizes = max(max_num_sizes, num_sizes)

        exp_dict = {exp['model']:exp for exp in ds_experiments}
        exp_names = sorted([exp['model'] for exp in ds_experiments],key=modelrank)
        results_group = {
            'dataset':ds,
            'metric': metric_transform(metric),
            'train_sizes':train_sizes,
            'train_sizes_str':train_sizes_str,
            'models':[]
        }
        for exp_name in exp_names:
            model_exp = exp_dict[exp_name]
            name = model_exp['model']
            results_str = []
            means = []
            stds = []

            met = exp['metrics'][metric]
            for i in range(num_sizes):
                met_result = '--'
                if 'means' in met:
                    mean = met['means'][i]
                    std = met['stds'][i]
                    met_result = f'{mean:.2f} \pm {std:.2f}'

                results_str.append(met_result)
                means.append(mean)
                stds.append(std)
            results_group['models'].append({
                'name':name,
                'results_str': results_str,
                'means':means,
                'stds':stds
            })
        
        for i in range(num_sizes):
            best_mean = 0
            best_idx = 0
            for j in range(len(exp_names)):
                mean = results_group['models'][j]['means'][i]
                if mean > best_mean:
                    best_mean = mean
                    best_idx = j
            old_txt = results_group['models'][best_idx]['results_str'][i]
            new_txt = f'$\mathbf{{{old_txt}}}$'
            results_group['models'][best_idx]['results_str'][i] = new_txt
            for j in range(len(exp_names)):
                if j != best_idx:
                    old_txt = results_group['models'][j]['results_str'][i]
                    new_txt = f'${old_txt}$'
                    results_group['models'][j]['results_str'][i] = new_txt

        
        result_groups.append(results_group)

    # add blanks to sizes as necessary
    for group in result_groups:
        curr_len = len(group['train_sizes_str'])
        for i in range(max_num_sizes - curr_len):
            group['train_sizes_str'].append('--')
        for model in group['models']:
            curr_len = len(model['results_str'])
            for i in range(max_num_sizes - curr_len):
                model['results_str'].append('')

    return result_groups

def generate_table(result_groups):
    n_sizes = len(result_groups[0]['train_sizes_str'])

    table = Tabular('l l l ' + 'c '*n_sizes, booktabs=True)
    table.add_row(
        MultiColumn(2, align='c', data='Task / Model'), 
        MultiColumn(n_sizes+1, align='c', data='Training Set Size')
    )
    table.add_hline()
    for group in result_groups:
        met_name = group['metric']
        multicol = MultiColumn(2, align='l', data=group['dataset'])
        cols = [multicol,NoEscape(f'{met_name}@:')] + group['train_sizes_str']
        table.add_row(cols)
        table.add_hline(start=1, end=2, cmidruleoption='r')
        table.add_hline(start=3, end=n_sizes+3, cmidruleoption='l')
        for model in group['models']:
            name = model['name']
            results = model['results_str']
            table.add_row('', name, '', *[NoEscape(x) for x in results])
    return table.dumps()


def test():
    experiments = [
        {
            'dataset':'Fusion360Seg',
            'model':'Geo',
            'F_1':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89],
                100000:[.9,.96,.9,.87]
            }
        },
        {
            'dataset':'Fusion360Seg',
            'model':'Geo+Topo',
            'F_1':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89],
                100000:[.9,.96,.9,.87]
            }
        },
        {
            'dataset':'Fusion360Seg',
            'model':'Topo',
            'F_1':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89],
                100000:[.9,.96,.9,.87]
            }
        },
        {
            'dataset':'Fusion360Seg',
            'model':'Geo+MP',
            'F_1':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89],
                100000:[.9,.96,.9,.87]
            }
        },
        {
            'dataset':'MFCAD',
            'model':'Geo',
            'F_1':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89]
            }
        },
        {
            'dataset':'MFCAD',
            'model':'Geo+Topo',
            'F_1':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89]
            }
        },
        {
            'dataset':'MFCAD',
            'model':'Topo',
            'F_1':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89]
            }
        },
        {
            'dataset':'MFCAD',
            'model':'Geo+MP',
            'F_1':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89]
            }
        },
        {
            'dataset':'Mating',
            'model':'Geo',
            'ap':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89]
            }
        },
        {
            'dataset':'Mating',
            'model':'Geo+Topo',
            'ap':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89]
            }
        },
        {
            'dataset':'Mating',
            'model':'Topo',
            'ap':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89]
            }
        },
        {
            'dataset':'Mating',
            'model':'Geo+MP',
            'ap':{
                10:[.4,.2,.34,.35],
                100:[.5,.52,.45,.55],
                1000:[.7,.6,.75,.7],
                10000:[.8,.81,.85,.89]
            }
        }
    ]

    result_groups = format_experiments(
        experiments, 
        preferred_metrics = {
            'Fusion360Seg': 'F_1'
        }, 
        metric_renames = {
            'F_1':'$F_1$',
            'ap':'AP'
        },
        model_order = ['Geo','Topo','Geo+Topo'],
        omitted_datasets=['MFCAD'],
        omitted_experiments=[('Fusion360Seg','Geo+Topo')],
        ommitted_models=['Geo'],
        placeholder_experiments=[('FabWave', 'Geo+Topo', 'ap', 6)],
        size_renames={1000:'1K', 10000:'10K',100000:'100K'}
    )

    for experiment in experiments:
        ds = experiment['dataset']
        model = experiment['model']
        with open(f'D:/btlpaper/data/example/{ds}_{model}.json','w') as f:
            json.dump(experiment, f)
    with open(f'D:/btlpaper/data/example/options.json','w') as f:
        json.dump(
            {
                'preferred_metrics':{
                    'Fusion360Seg': 'F_1'
                }, 
                'metric_renames': {
                    'F_1':'$F_1$',
                    'ap':'AP'
                },
                'model_order':['Geo','Topo','Geo+Topo'],
                'omitted_datasets':['MFCAD'],
                'omitted_experiments':[('Fusion360Seg','Geo+Topo')],
                'ommitted_models':['Geo'],
                'placeholder_experiments':[('FabWave', 'Geo+Topo', 'ap', 6)],
                'size_renames':{1000:'1K', 10000:'10K',100000:'100K'}
            }, 
        f)
    print(generate_table(result_groups))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--resdir', type=str)
    parser.add_argument('--outfile', type=str, default='')
    parser.add_argument('--options', type=str, default='')
    parser.add_argument('--print', action='store_true')
    args = parser.parse_args()
    if not os.path.exists(args.resdir):
        print(f'{args.resdir} -- directory not found')
        exit()
    if not os.path.isdir(args.resdir):
        print(f'{args.resdir} is not a directory')
        exit()
    experiment_files = [os.path.join(args.resdir,f) for f in os.listdir(args.resdir) if f.endswith('.json')]
    experiments = []
    for fname in experiment_files:
        with open(fname, 'r') as f:
            exp = json.load(f)
            if 'dataset' in exp and 'model' in exp:
                experiments.append(exp)
    if len(args.options) > 0:
        if os.path.exists(args.options):
            with open(args.options, 'r') as f:
                opts = json.load(f)
        else:
            print(f'{args.options} does not exist')
            exit()
    else:
        opts = dict()
    
    results_groups = format_experiments(experiments, **opts)
    latex = generate_table(results_groups)
    if args.print or len(args.outfile) == 0:
        print(latex)
    if len(args.outfile) > 0:
        outdir = os.path.dirname(args.outfile)
        os.makedirs(outdir, exist_ok=True)
        with open(args.outfile, 'w') as f:
            f.write(latex)
    