![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) [![DOI](https://zenodo.org/badge/612592716.svg)](https://zenodo.org/badge/latestdoi/612592716)

# Learning characteristics of graph neural networks predicting protein-ligand affinities

This repository contains the code for the work on protein-ligand interaction with GNNs, in which we investigate what GNNs learn by using explainable AI (XAI).

## Before you start
This code relies on the usage of [EdgeSHAPer](https://github.com/AndMastro/EdgeSHAPer) as an XAI method. However, there is no need to install it since all the files needed are provided in the ```src``` folder.

We suggest to run the code in a conda environment. We provide an ```environment.yml``` file that can be used to install the needed packages:

```bash
conda env create -f environment.yml
```

Note: this file was created from a conda working environment under Windows 11, so some packages may not be available in different versions/OS. We suggest to manually install them in this case. Edit the file with your system conda envs folder. We also provide the ```environment_no_builds.yml``` file for a higher cross-platform compatibility (minor manual edits to the file may be needed).

If you encounter problems during the installation of the PyTorch Geometric dependencies, manually install them in the conda environment using:

```bash
pip install torch_sparse==0.6.15 torch_cluster==1.6.0 torch_spline_conv==1.2.1 torch_geometric==2.1.0.post1 -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

Note that the versions above are the ones found in the ```environment.yml``` file. We suggest to install such version for reproducibility, but you are encouraged to try different versions to check for compatibility!

For compatibility with EdgeSHAPer code, [this](https://github.com/c-feldmann/rdkit_heatmaps) additional module must be installed. 

Finally, download the data we used in our experiments from [here](http://bioinfo-pharma.u-strasbg.fr/labwebsite/downloads/pdbbind.tgz) (copy and paste in a browser the link address if the download does not start) and nzipped them into the ```data```.

The config file ```parameters.yml``` contains settable parameters that are loaded by the scripts provided in the repo.

## Train the model

We provide the ```trainer_script.py``` file to train custom GNN models (check ```parameters.yml``` for details).

To train your model simply run:

```bash
python trainer_script.py
```
Alternatively, to reproduce the experiments, you can use the pretrained models available in ```models/pretrained_models``` folder.

## Explain the predictions

Using one of the pretrained models, or any custom model trained with ```trainer_script.py```, you can use ```explainer_script.py``` to run EdgeSHAPer to generate explanations in terms of important edges. Among other parameters found in ```parameters.yml```, ```AFFINITY_SET``` defines the affinity set for which to run the explanations (low, medium or high). To explain the predictions of the model defined by the ```MODEL_PATH``` parameter for the selected affinity set, simply run:

```bash
python explainer_script.py
```

Files contatining edge importance as Shapley value estimates by EdgeSHAPer will be saved into the folder specified by the ```SAVE_FOLDER``` parameters.

## Top-k edges computation

Once the explanations for all the affinity sets are obtained, it is possible to run:

```bash
python top_k_computation.py
```

to generate statistics and graphics for the top-k important edges of the samples explained. Additional parameters accepted can be found in the config file ```parameters.yml```.

A possible top-k edges (k=25) output image may look as:

<p align="center">
  <img src="results/explanations/GC_GNN/high affinity/5lwe/5lwe_EdgeSHAPer_top_25_edges_full_graph.png" alt="top-k edges for an example complex" width=45%>
</p>

The folder ```additional_scripts``` contains scripts for additional experiments and computations. A README file is provided in the folder with instructions.

### Contacts

For any questions, feel free to drop an [email](mailto:mastropietro@diag.uniroma1.it).

## Citations

If you use our work or results, please cite our publication in Nature Machine Intelligence 🧠

Mastropietro, A., Pasculli, G. & Bajorath, J. [Learning characteristics of graph neural networks predicting protein–ligand affinities](https://rdcu.be/dqZlS
). Nat Mach Intell (2023). https://doi.org/10.1038/s42256-023-00756-9 
