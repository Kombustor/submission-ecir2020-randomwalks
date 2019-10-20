# Reproduction Paper - Evaluation Code

This repository contains the evaluation code and datasets for the paper "Influence of Random Walk Parametrization on Graph Embeddings".

`visualization.ipynb` contains the reproduction of the Les Misérables case study (Figure 1 in the paper). In addition, t-sne visualizations are provided.  
`classification.ipynb` contains the reproduction of the node classification task, alongside with additional walk strategies and their evaluation (Table 1 in the paper).

## Setup

The easiest way is to setup a conda environment with the provided environment.yml:  
`conda env create -f environment.yml -n paper`  
Afterwards, activate the environment via `conda activate paper` and start a jupyter notebook server via `jupyter notebook`.

## Acknowledgement

The implementation is based on:
- https://github.com/aditya-grover/node2vec (node2vec reference implementation by the authors)

The datasets are taken from:
- http://konect.uni-koblenz.de/networks/moreno_lesmis (Les Misérables)
- http://socialcomputing.asu.edu/datasets/BlogCatalog3 (BlogCatalog)
