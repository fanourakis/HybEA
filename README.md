# HybEA: Hybrid Attention Models for Entity Alignment

## Getting Started

Install Anaconda 4.14.0

### Installation instructions

Install and activate the conda environment from .yml file using conda:

```bash
conda env create -f install/HybEA.yml
conda activate HybEA
```

### Generate entity name analytics/embeddings for name initialization

1. Choose dataset (D_W_15K_V1, D_W_15K_V2, SRPRS_D_W_15K_V1, SRPRS_D_W_15K_V2, BBC_DB, fr_en, ja_en, zh_en) from generate_names/Param.py (default is D_W_15K_V2)
2. Run HybEA/generate_names/main.py for running the name analysis, prioritization of names and generate the name embeddings.
```bash
cd generate_names
python main.py
```

### Run HybEA

1. Choose dataset (D_W_15K_V1, D_W_15K_V2, SRPRS_D_W_15K_V1, SRPRS_D_W_15K_V2, BBC_DB, fr_en, ja_en, zh_en) at attribute_model/Param.py
2. Choose dataset (D_W_15K_V1, D_W_15K_V2, SRPRS_D_W_15K_V1, SRPRS_D_W_15K_V2, BBC_DB, fr_en, ja_en, zh_en) at structure_model/Param.py
3. Choose dataset (D_W_15K_V1, D_W_15K_V2, SRPRS_D_W_15K_V1, SRPRS_D_W_15K_V2, BBC_DB, fr_en, ja_en, zh_en) at src/Param.py
4. Choose mode (Hybea, Hybea_light, Hybea_struct_first, Hybea_without_structure, Hybea_without_factual, Hybea_basic, Hybea_basic_structure_first) at src/Param.py for running experiments of HybEA and its' variations.
```bash
cd src
python main.py
```
5. In case of Hybea, Hybea_light, Hybea_struct_first, Hybea_basic, Hybea_basic_structure_first there is a need of calculation the final performance (see Calculate final performance instructions below).

### Calculate final performance of HybEA or its' variations

```bash
python calculate_performance.py [DATASET] [PATH TO FILES FOR NEW PAIRS] [HYBEA OR ITS' VARIATION]
# e.g.
python calculate_performance.py D_W_15K_V1 experiments/D_W_15K_V1_Hybea/ Hybea
```

### Calculate cummulative precision/recall/f1

```bash
python calculate_cummulative.py [DATASET] [PATH TO FILES FOR NEW PAIRS]
# e.g.
python calculate_cummulative.py D_W_15K_V1 experiments/D_W_15K_V1_Hybea/
```

### Knowformer Baseline

1. Choose dataset (D_W_15K_V1, D_W_15K_V2, SRPRS_D_W_15K_V1, SRPRS_D_W_15K_V2, BBC_DB, fr_en, ja_en, zh_en) at structure_model/Param.py
2. Set RANDOM_INITIALIZATION=True at structure_model/Param.py
3. Choose mode --> Hybea_without_factual at src/Param.py

```bash
cd src
python main.py
```

### RREA, BERT-INT, PipEA
For RREA and BERT-INT we used the code provided in [Knowledge Graph Embedding Methods for Entity Alignment: An Experimental Review](https://github.com/fanourakis/experimental-review-EA).
For PipEA we used the code of the original paper [PipEA](https://github.com/wyy-code/PipEA)

### Statistics of the datasets

```bash
cd statistics
python analysis.py
```
