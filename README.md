## Introduction
This repository contains the code for the ACL-21 paper:[Improving Lexically Constrained Neural Machine Translationwith Source-Conditioned Masked Span Prediction](https://arxiv.org/abs/2105.05498).

## Requirments and Installation
- [Pytorch](https://pytorch.org) version == 1.7.1
- Python version >= 3.7

##### Installing from source

To install fairseq from source and develop locally :
```
git clone https://github.com/wns823/NMT_SSP.git
cd NMT_SSP
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
pip install --editable .
```

## Getting Started

### Data preprocessing
In Ko-En dataset, 


In De-En datset,

dictionary matching
span making

### Train a transformer with SSP
First, 


### Generate
First, 


### Citation

```bibtex
@inproceedings{nmtssp2021,
  title     = {Improving Lexically Constrained Neural Machine Translation with Source-Conditioned Masked Span Prediction},
  author    = {Gyubok Lee, Seongjun Yang, Edward Choi},
  booktitle = {Proceedings of {ACL} 2021: Main conference},          
  pages     = {0--0},
  year      = {2021},
  month     = {8},
}
```
