## Experiment Workflow

Pipeline:

1. Filter dataset to valid, simple parts
2. Train Renderer
3. Train Tasks
  a. Fusion360Seg
  b. MFCAD
  c. Fabwave
  d. Automate
4. Evaluate Fitting Task


### Filter Dataset to Valid, Simple Parts

Not all part files (.x_t or .step) or supported by our pipeline. This could be because

a) the part file is invalid (does not not load in the kernel)
b) we get bad samples for the part
c) the part is not simple (complex face/edge types, or has too many edges adjacent to a face)

For any dataset we want to use, we need to filter out these parts ahead-of-time. Doing so for
reasons (a) and (c) are fairly straightforward, though we do have some parallelism issues.
Doing so for (b) is tricky since we don't yet know if we will always get bad samples given
the same part.

### Train Renderer

The renderer takes in a set of parts as ImplicitParts, and trains an encoder/decoder on them.
We may also optionally train a graph level encoder on them. This can produce two sets of codes
for each dataset.

### Train Tasks

Starting from a trained renderer, the various task ablations take either just a code, or a 
code and a simplified (face-face) adjacency structure. This means we start with a pre-processing
step for each dataset, that pre-computes the generated codes and extracts the graphs, and puts
those into a single pytorch tensor that we can fit in-memory at training time for very fast
training.

Each individual task is either a face or a part level task, and is either binary or multiclass
classification. In order to effectively share code, we should add module switches for number
of classes and for what kind of input to use / experiment to run, and then use the datasets
to do the appropriate samplings. Some of the tasks are not NN trained, and so just use SVMs,
so we will have two training scripts - SVM and NN.

After each training, it should store its best weights and test set scores in an appropriately
named json file for use with the paper generation code.

### Evaluate Fitting Task

We need a model and an edit to try this on


### Proposed Separation of Repositories

The parasolid frustrum, brep loading code, automate, and representation learning code
can and should be moved into separate repositories.

Proposed Separation:

parasolid: C++/CMake Project with just the frustrum code

brep_reader: C++/CMake Project with read_file for step/x_t (x_t optional) that has bodies,
but no parts

automate: Python/C++ project. Contains SBGCN and related gcn code,
Part from pspy, and the automate tool

repbrep/fsbrep/hybridrep: Python/C++ Project. Contains implicit_part,
plus any training and evaluation code. Has index jsons committed, and
download scripts to download zips for other datasets. Has code to
make plots from parquet files, and helper scripts to generate those.