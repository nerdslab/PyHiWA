# Hierarchical Wasserstein Alignment (HiWA)
--------------------------------------------
This repository contains a Python package implementing the algorithm described in this paper:

John Lee, Max Dabagia, E Dyer, C Rozell: Hierarchical Wasserstein Alignment for Multimodal Distributions, NeurIPS 2019. https://arxiv.org/abs/1906.11768

## Overview
----------
Optimal transport approaches to distribution alignment attempt to minimize the divergence between two distributions, as quantified by the Wasserstein distance between them. HiWA introduces some further assumptions which make this problem tractable even in the presence of noise and ambiguity, which are unavoidable in real-world datasets. These assumptions are:
  1. Well-defined cluster structure exists in each dataset.
  2. The inter- and intra-cluster structure is consistent between datasets.
HiWA leverages this cluster structure by first determining how best to align the clusters, and then using this information to influence a aligning transformation of the entire dataset to match the target.

## Contents
----------
The `HiWA` class in this repository is a self-contained implementation of the algorithm. The included Jupyter Notebook is a comprehensive demo with the algorithm applied to both a synthetic dataset and a real-world neuroscience problem on decoding movement intention from neuron firing patterns in the primary motor cortex of a non-human primate. 

## Dependencies
---------------
`numpy, scipy, matplotlib, scikit-learn`

## Tips
-------
- The dimensionality reduction technique used to construct the low-dimensional embedding is *critically* important to the algorithm's success. The first thing to check if it is not working is whether the low-dimensional embeddings of the source and target datasets are capturing the same structure, and whether that structure is alignable (i.e. there are no pathological symmetries).
