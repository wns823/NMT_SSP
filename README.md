## Introduction
This repository contains the code for the ACL-21 paper: 

[Improving Lexically Constrained Neural Machine Translation with Source-Conditioned Masked Span Prediction](https://arxiv.org/abs/2105.05498)

Gyubok Lee*, Seongjun Yang*, Edward Choi (\*: equal contribution)

Our code is built upon [Leca](https://github.com/ghchen18/leca) because it works for both with or without a bilingual term dictionary. 

## Data download
- DE-EN OPUS Acquis, Emea <br>
Download the DE-EN OPUS Acquis, Emea dataset by this [link](https://opus.nlpl.eu/)

- DE-EN IATE dictionary <br>
Download the DE-EN dictionary by this [link](https://iate.europa.eu/)

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
pip install parmap
```

## Getting Started

### 1. Data preprocessing
#### Step1. Place the data in the appropriate folder.
```
dict_law_en_ko.json, iate_en_de_all.json → dictionary folder
corpus → raw_data folder
```

#### Step2. Filter dictionary in IATE dictionary. (filter_dict.py)
```bash
python filter_dict.py --min 4 --max 20
```
#### Step3. Split data by terminology-aware data split algorithm (data_split_algorithm.py)
```bash
ex) python data_split_algorithm.py --domain acquis --src_path raw_data/JRC-Acquis.de-en.de  --tgt_path raw_data/JRC-Acquis.de-en.en --directory_path dictionary/iate_en_de_filter.json --src_lang de
ex) python data_split_algorithm.py --domain emea --src_path raw_data/EMEA.de-en.de  --tgt_path raw_data/EMEA.de-en.en --directory_path dictionary/iate_en_de_filter.json --src_lang de
ex) python data_split_algorithm.py --domain law --src_path raw_data/law-all.ko  --tgt_path raw_data/law-all.en --directory_path dictionary/dict_law_en_ko.json --src_lang ko
```

#### Step4. Tokenize and BPE
```bash
In DE-EN,
bash tokenizing_bpe_gen.sh domain
ex) bash tokenizing_bpe_gen.sh acquis
bash tokenizing_bpe_apply.sh domain split
ex) bash tokenizing_bpe_apply.sh acquis valid
In KO-EN,
bash tokenizing_bpe_gen_ko.sh
bash tokenizing_bpe_apply_ko.sh
```

#### Step5. Filter sentence by length (filter_data.py)
```bash
ex) python filter_data.py --domain emea --src_lang de --tgt_lang en --split train --min 5 --max 80
ex) python filter_data.py --domain emea --src_lang de --tgt_lang en --split valid --min 5 --max 80
ex) python filter_data.py --domain emea --src_lang de --tgt_lang en --split test --min 5 --max 80
```
#### Step6. Binarize dataset (binarize_dataset.sh)
```bash
bash binarize_dataset.sh emea de en
```

#### Step7. Span making (make_tok.sh -> make_span.py)
```bash
ex)
bash make_tok.sh de en emea
python make_span.py --directory emea_deen --src de --tgt en --saved data-bin/emea_deen
```

### 2. Train a transformer with SSP
```bash
bash train.sh gpu domain src tgt model_path span loss_ratio min_span max_span dropout
ex) bash train.sh 2 emea de en emea_leca_span_0.3 span 0.5 1 10 0.3
```


### 3. Generate
```bash
bash inference.sh gpu domain src tgt model_path with_dictionary
ex) bash inference.sh 0 emea de en emea_leca_span_0.3 1 (without dictionary)
ex) bash inference.sh 0 emea de en emea_leca_span_0.3 0 (with dictionary)
```

### 4. TER, LSM score
```bash
ex) python ngram_inference.py --domain emea --src_lang de --tgt_lang en --outputfile inference_result/emea_leca_span_0.3_1.txt
```


### Citation

```bibtex
@inproceedings{lee-etal-2021-improving,
    title = "Improving Lexically Constrained Neural Machine Translation with Source-Conditioned Masked Span Prediction",
    author = "Lee, Gyubok  and
      Yang, Seongjun  and
      Choi, Edward",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.94",
    doi = "10.18653/v1/2021.acl-short.94",
    pages = "743--753",
    abstract = "Accurate terminology translation is crucial for ensuring the practicality and reliability of neural machine translation (NMT) systems. To address this, lexically constrained NMT explores various methods to ensure pre-specified words and phrases appear in the translation output. However, in many cases, those methods are studied on general domain corpora, where the terms are mostly uni- and bi-grams ({\textgreater}98{\%}). In this paper, we instead tackle a more challenging setup consisting of domain-specific corpora with much longer n-gram and highly specialized terms. Inspired by the recent success of masked span prediction models, we propose a simple and effective training strategy that achieves consistent improvements on both terminology and sentence-level translation for three domain-specific corpora in two language pairs.",
}
```
