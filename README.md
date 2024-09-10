<p align="center">
  <img src="/assets/n3mdl_preliminary_logo.png">
</p>

<p align="center">

[![LICENSE](https://img.shields.io/badge/license-MIT-brightgreen)](https://github.com/dean-jordan/N3MDL-Symbolic-Networks/blob/main/LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/dean-jordan/N3MDL-Symbolic-Networks)

</p>

<h3 align="center"> N3MDL: Efficient Multi-Domain Learning with Neurosymbolic Neural Network Ensembles <h3>

N3MDL aims to improve Deep Learning performance for Multi-Domain tasks through a novel Neural Network architecture resembling the human brain more closely.

---
- [About](#about)
- [Quickstart](#quickstart)
- [N3MDL Source](#n3mdl-source)
    - [Activation](#activation)
    - [Adapters](#adapters)
    - [Attention](#attention)
    - [Encoder](#encoder)
    - [Decoder](#decoder)
    - [Layers](#layers)
    - [Loss](#loss)
    - [Network](#network)
    - [Symbolic](#symbolic)
- [Models](#models)
- [Docs](#docs)
- [Notebooks](#notebooks)
- [Reports](#reports)
- [References](#references)
- [Training](#training)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)
---

### About

### Quickstart
Quickstart for cloning repository and either running an instance of the model or making a commit to the repository.

```bash
# Clone Repository
git clone https://github.com/dean-jordan/N3MDL-Symbolic-Networks.git

# Prerequisite Installation
pip3 install -r requirements.txt
```

For modifying source, please utilize the N3MDL directory.

For running model and making predictions:

```bash
# Running Final Model
python3 ./models/N3MDL_model.py
```

Ensure that all commands are running from the root directory (`N3MDL-Symbolic-Networks`) in order to utilize relative pathing in model commands.

The N3MDL software package was developed on Linux Ubuntu with Python v3.11.5 and Amazon Linux 2023 with Python v3.11.5.
Any Python 3.x version is compatible. However, newer versions of Python are recommended for compatibility.

### N3MDL Source
The N3MDL network definition is divided into nine categories. However, for quick access, the `./models/N3MDL_model.py` file contains a script allowing for access of the full model to run locally.

All other files were used for the creation, training, and evaluation of the model. All directories and key files will be described below.

#### Activation
The activation directory contains all activation functions which will be used to intiialize every part of the network. This directory contains existing and custom activation functions.

`custom.py`
> Contains custom activation functions for initialization of smaller network modules.

#### Adapters
The adapters directory contains files for the creation and access of adapters. This allows for the model to be specialized on-the-fly and perform complex multi-domain operations without requiring excessive RAM.

`adapter_creation.py`
> Contains code for the creation of adapters within the training process.

`adapter_loading.py`
> Contains code to dynamically load adapters upon the usage of the model.

`dynamic_adapter_creation.py`
> Contains code for continual learning. Allows for adapters to be created as the model is being used for increased specialization and personalization.

#### Attention
The attention directory contains modules for the development of a novel attention mechanism for N3MDL. This has a variety of advantages discussed in the publication.

`attention.py`
> Creates attention mechanism through `query.py`, `value.py`, and `weight.py`.

`mask.py`
> Creates masking in the attention mechanism to prevent decoder collapse during training.

#### Encoder
The encoder directory produces a neural network module which separates input into several outputs. The module itself produces low-level features in order for further processing by the middle subnetwork ensemble.

`input_embedding.py`
> Uses attention mechanism to tokenize inputs.

`encoder.py`
> Creates encoder module through attention mechanism (`encoder_attention.py`) and encoder blocks (`encoder_block.py`).

#### Decoder
The decoder takes the input of the neural subnetwork ensemble and translates the low-level features back into a full output. The directory defines the code for a decoder block and a decoder module.

`decoder_block.py`
> Defines a single decoder block. This can then be used to create a full decoder mechanism.

`decoder.py`
> Allows for decoder to be defined by combining decoder blocks and organizing them to produce final output.

#### Layers
The layers directory defines many custom layers used throughout the project. Each layer is named with the part of the network it corresponds to and its number defines where in a block the layer is placed.

`integration.py`
> Defines an integration layer for Neurosymbolic architecture.

#### Loss
Due to the novelty of the neural network architecture, a custom loss function is utilized. This directory defines the function and creates plotting for the function in order to test its capability.

`loss.py`
> Defines the custom loss function.

#### Network
Defines the subnetwork ensemble that occurs between the encoder and the decoder. This processes the low-level inputs and each subnetwork is assigned to an adapter, allowing for specialization and generalization to be balanced.

`subnetwork.py`
> Initializes a single subnetwork in the overall ensemble.

`network.py`
> Initializes the full subnetwork ensemble by combining subnetworks in a specific organization.

#### Symbolic
Because the architecture takes advantage of Neurosymbolic programming in order to allow for logical tasks, the Symbolic directory allows for the Symbolic principles to be realized.

`symbolic_reasoning_engine.py`
> Creates a framework for the network to deduce proofs and generate knowledge.

`symbolic.py`
> Creates a symbolic module through combining all parts of the symbolic system together.

### Models
The models directory contains the fully-trained model in multiple formats. PyTorch state dictionaries can be used, but [ONNX](https://onnx.ai) is the preferred method of accessing the serialized model.

`N3MDL_model.py`
> Contains a script for accessing and communicating with the ONNX-serialized model.

### Docs
Contains documentation detailing user and developer workflows. For more information, see the `./docs` directory.

### Notebooks
Contains notebooks detailing how the model was developed. Each notebook is numbered in order to define a simple workflow. Note: if a commit is being made to the code, it is expected that the notebooks are updated accordingly.

### Reports
Contains the final report. This is written in the TeX format, but a finalized PDF-based copy is contained within the `./reports` directory.

### References
Contains all references which are cited in the report. This is contained in the BibTeX format.

### Training
Training occurs through a custom training algorithm defined in the train.py directory. Please refer to the `./training` directory for the scripts and code used to train the model.
Note that this directory does not contain the full dataset as described in the paper. However, the dataset is open-access.

### Dependencies
- Python 3.11.5
    - torch
    - matplotlib
    - numpy
    - onnxruntime

### Directory Structure

```
├── LICENSE            <- MIT Open-source license
├── Makefile           <- Makefile allowing for commands such as `make data` and `make train`
├── README.md          <- The top-level README for an overview of N3MDL
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── docs               <- A mkdocs-based documentation system detailing usage of the project
│
├── models             <- Fully trained and serialized models, including ONNX and PyTorch checkpoints
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         N3MDL and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── N3MDL   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes N3MDL a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

