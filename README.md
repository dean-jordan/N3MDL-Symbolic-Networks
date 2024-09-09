<p align="center">

[![LICENSE](https://img.shields.io/badge/license-MIT-brightgreen)](https://github.com/dean-jordan/N3MDL-Symbolic-Networks/blob/main/LICENSE)

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

### N3MDL Source

#### Activation

#### Adapters

#### Attention

#### Encoder

#### Decoder

#### Layers

#### Loss

#### Network

#### Symbolic

### Models

### Docs

### Notebooks

### Reports

### References

### Training

### Dependencies

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

