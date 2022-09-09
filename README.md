# SAT Benchmark

## Introduction

**SAT** **B**enchmark (`satb`) is a PyTorch implementation of a collection of published end-to-end SAT solvers with the ability to reproduce results claimed in the original papers. Hope we can build a comprehensive and practical codebase like [timm](https://github.com/rwightman/pytorch-image-models), and boost the AI4SAT research.

**All contributions (no matter if small) are always welcome!!!**


## Models

1. [NeuroSAT](https://arxiv.org/abs/1802.03685)/[NeuroCore](https://arxiv.org/abs/1903.04671)
2. [DG-DAGRNN](https://openreview.net/forum?id=BJxgz2R9t7)
3. [SATformer](https://arxiv.org/abs/2209.00953)
4. [DeepSAT](https://arxiv.org/abs/2205.13745)
5. [QuerySAT](https://arxiv.org/abs/2106.07162)
6. [FourierSAT](https://arxiv.org/abs/1912.01032)
7. *TBD*


## Datasets

We generate the common datasets used in end-to-end SAT training/inference. Various formats include CNF, Circuit by `cube and conquer`, and Circuit by [abc](https://github.com/berkeley-abc/abc) optimization. For more details, see [DATA.md](doc/DATA.md)

1. SR3-10
2. SR10-40 
3. SR3-100
4. Graph
   * coloring
   * dominating set
   * clique detection
   * vertex cover
5. Logic equivalent checking
6. SAT-Sweeping
7. SHA-1 preimage attack

## Features

Following list some features in my mind that should be realized in our codebase. Some features are borrowed from [timm](https://github.com/rwightman/pytorch-image-models) directly.
* All models have a common default configuration interface and API for 
   * doing a forward pass on just features (or recursively) `forward_features`. The format of features depends on the representation of problems, e.g., CNFs, Circuits, etc.
   * accessing the solver module `decode_assignment` to decode the assignemnts.
* High performance [reference training, validation, and inference scripts](scripts) that work in several process/GPU modes:
   * NVIDIA DDP w/ a single GPU per process, multiple processes with APEX present (AMP mixed-precision optional)
   * PyTorch DistributedDataParallel w/ multi-gpu, single process (AMP disabled as it crashes when enabled)
   * PyTorch w/ single GPU single process (AMP optional)

