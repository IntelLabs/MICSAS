# MISIM: A Neural Code Semantics Similarity System Using the Context-Aware Semantics Structure

MISIM is a neural code semantics similarity system that introduces a novel code representation named context-aware semantics structure (CASS in short) and a neural-backend that supports various neural network architectures. 

Further details can be found in the [technical paper](https://arxiv.org/pdf/2006.05265.pdf) titled "A Neural Code Semantics Similarity System Using the Context-Aware Semantics Structure" by Fangke Ye, Shengtian Zhou, Anand Venkat, Ryan Marcus, Nesime Tatbul, Jesmin Jahan Tithi, Niranjan Hasabnis, Paul Petersen, Timothy Mattson, Tim Kraska, Pradeep Dubey, Vivek Sarkar, Justin Gottschlich.

## Requirements

- Python 3.7.6
- Python packages
    * absl-py 0.9.0
    * numpy 1.18.1
    * pyprg 0.1.1b7
    * regex 2020.4.4
    * scipy 1.4.1
    * sklearn 0.0
    * torch 1.6.0
    * torch-scatter 2.0.5
    * tqdm 4.42.1
    * tree-sitter 0.1.1
    * wget 3.2
    * networkx 2.4
- CMake
- C++14 compatible compiler
- Clang++ 3.7.1 (optional, for the preprocessing step of Neural Code Comprehension)

## Training

1. Data preprocessing
    - Run `./preprocess.sh` (clang++-3.7.1 required), or
    - Download preprocessed datasets from [here](https://www.dropbox.com/s/zilq32a4s9pygde/datasets.tar.xz) and extract them into `data/`.

2. Training

    Use the commands below to train each model described in the paper:
    ```
    python train.py <model_name> --split data/datasets/split_<dataset_name>.pkl -f data/datasets/<dataset_name>/dataset-<model_name> --save data/models/<dataset_name>/<model_name>
    ```

    `<dataset_name>` can be one of:
    - poj (POJ-104)
    - gcj (Google Code Jam)

    `<model_name>` can be one of:
    - gnn (MISIM-GNN)
    - sbt (MISIM-RNN)
    - bof (MISIM-BoF)
    - c2v (code2vec)
    - ncc (Neural Code Comprehension with inst2vec)
    
    To train the Neural Code Comprehension model without inst2vec, use the following command:
    ```
    python train.py ncc -noi2v --split data/datasets/split_<dataset_name>.pkl -f data/datasets/<dataset_name>/dataset-ncc --save data/models/<dataset_name>/ncc-noi2v
    ```

    Pre-trained models are available [here](https://www.dropbox.com/s/jlfp2oypzkc29q7/models.tar.xz). They include the models trained with three different random seeds, and were used to obtain the evaluation results in the paper.

## Evaluation

For deep learninig models including MISIM, code2vec, and Neural Code Comprehension, use the following command to compute and print the evaluation metrics (`<model_file>` is the name of the model file (`model.pt`) obtained from training):
```
python train.py <model_name> --split data/datasets/split_<dataset_name>.pkl -f data/datasets/<dataset_name>/dataset-<model_name> --load <model_file>
```

To evaluate Aroma, use the following commands:
```
# Aroma-Dot
python aroma.py -f data/datasets/<dataset_name>/dataset-aroma --split data/datasets/split_<dataset_name>.pkl --sim dot

# Aroma-Cos
python aroma.py -f data/datasets/<dataset_name>/dataset-aroma --split data/datasets/split_<dataset_name>.pkl --sim cos
```
