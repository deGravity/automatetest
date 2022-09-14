import numpy as np
import altair as alt

def plot_accuracies(frame, dataset):
    line = alt.Chart(frame).mark_line().encode(
        x=alt.X('train_size',scale=alt.Scale(type="log")),
        y=alt.Y('mean(accuracy)'),
        color='model'
    )

    band = alt.Chart(frame).mark_errorband(extent='ci').encode(
        x=alt.X('train_size', scale=alt.Scale(type="log")),
        y=alt.Y('accuracy', title='accuracy'),
        color=alt.Color('model')
    )

    return (band + line).properties(
        title=f'{dataset} Accuracy vs Train Size'
    )

def plot_segmentation_accuracies(frame, dataset, average='micro'):
    frame = frame[frame.dataset == dataset]
    if average=='macro':
        frame = frame.groupby(
            ['dataset','model','train_size','seed','test_idx']
            ).agg({'accuracy':'mean'}).reset_index()
    frame = frame.groupby(
        ['dataset','model','train_size','seed']).agg({'accuracy':'mean'}).reset_index()
    return plot_accuracies(frame, dataset)

def plot_classification_accuracies(frame, dataset, size_proto_model='Ours'):
    ts_frac = frame[frame.model==size_proto_model].groupby('train_fraction').agg({'train_size':'min'}).reset_index()
    tf_to_ts = dict(zip(ts_frac.train_fraction,ts_frac.train_size))
    frame.train_size = np.vectorize(lambda x: tf_to_ts[x])(frame.train_fraction)
    frame = frame.groupby(['dataset','model','train_size','seed']).agg({'accuracy':'mean'}).reset_index()
    return plot_accuracies(frame, dataset)

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