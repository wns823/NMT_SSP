## Introduction
This repository contains the code for the ACL-21 paper:[Improving Lexically Constrained Neural Machine Translationwith Source-Conditioned Masked Span Prediction](https://arxiv.org/abs/2105.05498).

We show code that applies our method to [Leca](https://github.com/ghchen18/leca) because it can show inference results without dictionary and with dictionary. 

## Data download
I'll uploading data and data link soon.
- DE-EN OPUS Acquis, Emea <br>
Download the DE-EN OPUS Acquis, Emea dataset by this [link](https://opus.nlpl.eu/)

- DE-EN IATE dictionary <br>
Download the DE-EN dictionary by this [link](https://drive.google.com/file/d/1XFJ257xK3eAzh9tRnJMGm0KCRl3TyJr9/view?usp=sharing)

- KO-EN Law data <br>
Download the KO-EN corpus by this [link](https://www.aihub.or.kr/aidata/87/download)
- KO-EN Law dictionary <br>
Download the KO-EN dictionary by this [link](https://drive.google.com/file/d/1n626huC-6x5R7OEzLiKr5N7ulNGMxrLJ/view?usp=sharing)

## Requirments and Installation
- [Pytorch](https://pytorch.org) version == 1.7.1
- Python version >= 3.7

##### 0. Installing from source

To install fairseq from source and develop locally :
```
git clone https://github.com/wns823/NMT_SSP.git
cd NMT_SSP
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
pip install --editable .
pip install wandb
pip install spacy==2.2.4
pip install mecab-python3==0.996.5
pip install konlpy==0.5.2
pip install tokenizers==0.10.2
```

## Getting Started

### 1. Data preprocessing
Step1. Place the data in the appropriate folder.

Step2. Filter dictionary in IATE dictionary. (filter_dict.py)

Step3. Split data by terminology-aware data split algorithm (data_split_algorithm.py)

Step4. Delete sentences that doesn't match the dictionary

Step5. Filter sentence by length (filter_data.py)

Step6. Binarize dataset (binarize_dataset.sh)

Step7. Span making (make_tok.sh -> make_span.py)

### 2. Train a transformer with SSP
```bash
bash train.sh gpu_number domain src tgt model_path span loss_ratio min_span max_span dropout
ex) bash train.sh 3 acquis de en acquis_leca_span_0.3 span 0.5 1 10 0.3
```


### 3. Generate
```bash
bash train.sh gpu_number domain src tgt model_path use_dictionary
ex) bash inference.sh 0 acquis de en acquis_leca_span_0.3 0
```

### 4. TER, LSM score
```bash
ex) python ngram_inference.py --domain acquis --src_lang de --tgt_lang en --outputfile result_collection/law_leca_span_with_dict.txt
```


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
