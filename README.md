# Spreading vectors for similarity search

This is the open source implementation of the neural Catalyzer for similarity search.
This code reproduces the results from the ICLR'2019 paper Spreading Vectors for Similarity Search.


## Install

The basic install only requires Numpy and Pytorch 1.0
```python
conda install numpy
# See http://pytorch.org for details
conda install pytorch -c pytorch
```


This code can run as is on a standard computer, but it detects if a GPU is present and automatically uses it.

### (optional) GPU Faiss

If you want to further accelerate the code, you can install [Faiss](https://github.com/facebookresearch/faiss) with GPU support:
```bash
# Make sure you have CUDA installed before installing faiss-gpu, otherwise it falls back to CPU version
conda install faiss-gpu -c pytorch # [DEFAULT]For CUDA8.0
conda install faiss-gpu cuda90 -c pytorch # For CUDA9.0
conda install faiss-gpu cuda92 -c pytorch # For CUDA9.2
```

### (optional) Install the C lattice quantizer

The lattice quantizer can be run much faster custom C extensions.
We provide a C implementation of the lattice quantizer, wrapped in Python using SWIG.
First, you need to download and install Swig from your system's package manager or from [the website](http://www.swig.org/download.html).

The C code can then be compiled:
```
cd lattices
make all
```


## Evaluating a model

To benchmark our method, we use the two standard benchmark datasets [BigANN](http://corpus-texmex.irisa.fr/) and [Deep1b](https://yadi.sk/d/11eDCm7Dsn9GA), see [here](https://github.com/facebookresearch/faiss/tree/master/benchs#getting-bigann) for more info on how to download.
You need to indicate the path to these in lib/data.py:
```python
# lib/data.py
def getBasedir(s):
    paths = {
        "bigann": "/path/to/bigann",
        "deep1b": "/path/to/deep1b"
    }

    return paths[s]
```
Note that for both Bigann an Deep1b, only the first 1M vectors of the dataset are used (hence they are called Deep1M and Bigann1M in the paper).

```
python eval.py --ckpt test.pth --quantizer zn_79
```

### Pre-trained models

We provide pre-trained models.
The script [reproduce.sh](reproduce.sh) downloads the models and reproduces the paper's main results.

## Training a model

Run training:
```
python train.py --num_learn 500000 --database bigann --lambda_uniform 0.02 --dint 1024 --dout 24
```

Typical output:
```
load dataset deep1b
keeping 500000/357380000 training vectors
computing training ground truth
build network
Lr schedule ...
  Forward pass
  Distances
  Train
epoch 0, times: [hn 3.99 s epoch 55.19 s val 0.00 s] lr = 0.100000 loss = -0.00585795 = 0.00175652 + lam * -3.80723, offending 17773
  Forward pass
  Distances
  Train
epoch 1, times: [hn 4.07 s epoch 57.41 s val 0.00 s] lr = 0.100000 loss = -0.0034838 = 0.00245264 + lam * -2.96822, offending 56211
  Forward pass
  Distances
  Train

....

  epoch 8, times: [hn 4.04 s epoch 55.10 s val 0.00 s] lr = 0.100000 loss = -0.00382894 = 0.00203354 + lam * -2.93124, offending 75412
  Forward pass
  Distances
  Train
Valiation at epoch 9
zn_3     nbit= 14:  0.0000 0.0003 0.0028
zn_10    nbit= 32:  0.0009 0.0073 0.0437
zn_79    nbit= 64:  0.0331 0.1581 0.4756
storing test_ckpt/0.002/checkpoint.pth
zn_79,rank=10 score improves (0.15814 > 0), keeping as best
```

Training uses a small part of the learn set, split between 500k training vectors and 1M+10k validation vectors (1M database, 10k queries). The rest of the data is unused.

The ground-truth nearest neighbors are computed for the 500k vectors, this is fast enough on GPU.

The stats that are logged are:
- `lr`: the current learning rate, depends on the type of schedule
- `loss`: total_loss = triplet_loss + lambda * entropy_loss.
- `offending`: number of triplets that caused a non-0 loss (should be decreasing)
- `times`: hard-negative mining time, training time and validation time.

Validation is performed every 10 epochs (by default).
For a few quantizers (selected with --quantizers) it performs the search on a validation set and reports the 1-recalls at ranks 1, 10, 100.
Then it keeps the best model based on one of the evalated quantizers (zn_79 in this case: the Zn lattice with r^2 = 79).

Training for 160 epochs takes less than 3 hours on a P-100 GPU, but 90% of the final performance should already be reached in only 10 epochs (around 10 minutes).

### Cross-validation

The only parameter that we cross-validated is the lambda.
The script [crossvalidate.sh](crossvalidate.sh) does the grid search and tests on the best result.  
It runs the grid search sequentially, for a faster result it is worthwhile to run it on a cluster of machines.
A typical output is [this gist](https://gist.github.com/mdouze/bd34ceb6b17c3616e0b4e6a45e387cb7), which corresponds to the line "Catalyzer + lattice" of table 1 in the paper.

## Zn quantizer

The spherical Zn quantizer uses as codebook the points of the hypersphere of radius r that have integer coordinates. We provide here the common (squared) radiuses that correspond to 16, 32, and 64 bits for commonly used dimensions.


| d  | 16 bits | 32 bits | 64 bits |
|----|--------:|--------:|--------:|
| 24 |       3 |      10 |      79 |
| 32 |       3 |       8 |      36 |
| 40 |       2 |       7 |      24 |

To find out the number of bits needed to encode the vertices of a sphere in 48 dim with squared radius 30, use:
```python
from lattices.Zn_lattice import ZnCodec
import math

d, r2 = 48, 30
# number of distinct vertices
nv = ZnCodec(d, r2).nv

# number of bits needed to encode this
nbit = math.ceil(math.log2(nv))
```
In this case, nv = 311097167722066085728512 and nbit = 79.

### Search performance

A typical use case for the lattice quantizer is to perform asymmetric nearest-neighbors searches.
In that case, a set of n vectors is encoded.
At search time, a query vector x is compared to each of the encoded vectors.
Thus, each vector is decoded to y and the distance to x is computed.
The nearest vector id is then returned.
In general, this is done simultaneously for a batch of query vectors.

The benchmark [bench_Zn_decoder.py](lattices/bench_Zn_decoder.py) performs this operation, for 1M database vectors and 1k queries.
It compares the performance with that of a Faiss PQ index.
Typical result in [this gist](https://gist.github.com/mdouze/0b3ae8c88ba62aae234cdb8507164934): the lattice decoder is a bit slower than PQ.
This is understandable because PQ performs comparisons in the compressed domain.


## Adding datasets

We provide dataloaders for the standard BigANN and Deep1b.
Our code can easily be extended to other datasets.
First, add the path to your dataset in lib/data.py:
```python
# lib/data.py
def getBasedir(s):
    paths = {
        "bigann": "/path/to/bigann",
        "deep1b": "/path/to/deep1b",
        "my_data": "/path/to/mydata"
    }

    return paths[s]
```

Then, modify lib/data.py to handle the loading of your dataset
```python
#lib/data.py

def load_mydata(device, size = 10 ** 6, test=True, qsize=10 ** 5):
    basedir = getBasedir("my_data")

    # Exemple code to load your data
    xt = np.load(join(basedir, "my_trainingset.npy"))
    if test:
      xb = np.load(join(basedir, "my_database.npy"))
      xq = np.load(join(basedir, "my_queries.npy"))
    else:
        xb = xt[:size]
        xq = xt[size:size+qsize]
        xt = xt[size+qsize:]
        gt = get_nearestneighbors(xq, xb, 100, device)

    return xt, xb, xq, gt


def load_dataset(name, device, size=10**6, test=True):
    # ...
    elif name == "my_data":
      load_mydata(device, name, size, test)

```


## License

This repository is licensed under the CC BY-NC 4.0.
