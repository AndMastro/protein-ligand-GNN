![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) 

# Predicting affinities from simplistic protein-ligand interaction representations – what do graph neural networks learn?

This repository contains the code for the work on protein ligand interaction with GNNs, in which we investigate what GNNs learn by using explainable AI (XAI).

## Before you start
This code relies on the usage of [EdgeSHAPer](https://github.com/AndMastro/EdgeSHAPer) as an XAI method. However, there is no need to install it since all the files needed are provided in the ```src``` folder.

We suggest to run the code in a conda environment. We provide an environment.yml file that can be used to install the needed packages.

```bash
conda env create -f environment.yml
```
Note: this file was created from a conda working environment under Windows 11, so some packages may not be available in different versions/OS. We suggest to manually install them in this case. Edit the file with your system conda envs folder.

For compatibility with EdgeSHAPer code, [this](https://github.com/c-feldmann/rdkit_heatmaps) additional module should be installed. 

Finally, dowload the data we used in our experiments from [here](http://bioinfo-pharma.u-strasbg.fr/labwebsite/downloads/pdbbind.tgz) and unzipped them into the ```data```.
