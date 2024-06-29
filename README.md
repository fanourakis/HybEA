# HybEA: Hybrid Attention Models for Entity Alignment

The source code of the paper:
N. Fanourakis, F. Lekbour, V. Efthymiou, G. Renton, V. Christophides: HybEA: Hybrid Attention Models for Entity Alignment

## Dataset License

Due to licensing we are not allowed to distribute the dataset bbc-db.
To run the experiments, please download [bbc-db](https://www.csd.uoc.gr/~vefthym/minoanER/datasets.html)

## Getting Started

1. Install Anaconda 4.14.0

### Installation instructions

Install and activate the conda environment from .yml file using conda:

```bash
conda env create -f install/HybEA.yml
conda activate HybEA
```

### Generate entity name analytics/embeddings for name initialization

You should run the main.py of HybEA/generate_names for running the name analysis, prioritization of names and generate the name embeddings.
At first, please choose dataset from generate_names/Param.py (default is D_W_15K_V1)
```bash
cd generate_names
python main.py
```

### Run HybEA variations

#### HybEA
```bash

```

#### HybEA (struct. first)
```bash

```

#### HybEA (w/o struct.)
```bash

```

#### HybEA (w/o fact.)
```bash

```

#### HybEA (basic)
```bash

```

#### HybEA (basic; str.first)
```bash

```


### Knowformer Baseline

```bash
```

### Calculate cummulative precision/recall/f1

```bash
python calculate_cummulative.py [DATASET] [PATH TO FILES FOR NEW PAIRS]
# e.g.
python calculate_cummulative.py D_W_15K_V1 experiments/D_W_15K_V1_HybEA/
```

### RREA, BERT-INT, PipEA
For RREA and BERT-INT we used the code provided in [Knowledge Graph Embedding Methods for Entity Alignment: An Experimental Review](https://github.com/fanourakis/experimental-review-EA).
For PipEA we used the code of the original paper [PipEA](https://github.com/wyy-code/PipEA)

### Statistics of the datasets

```bash
cd statistics
python main.py
```