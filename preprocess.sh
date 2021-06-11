#!/bin/bash

# Build the CASS extractor
cd cass-extractor
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ../..

mkdir data/datasets

git clone https://github.com/spcl/ncc.git ncc/ncc

DATASETS=(poj gcj)
for DS in ${DATASETS[@]}; do

tar -xf data/${DS}_filtered.tar.bz2 -C data

# Split the dataset
python split_dataset.py -i data/${DS}_filtered -m ${DS} -o data/datasets/split_${DS}.pkl

# Extract CASSes
python extract.py -i data/${DS}_filtered -o data/${DS}_cass

# Extract AST paths for code2vec
python code2vec/extract.py -i data/${DS}_filtered -o data/${DS}_c2v -d ${DS}

# Prepare LLVM IR datasets for Neural Code Comprehension (NCC)
if [[ "$DS" == "poj" ]]; then
AUG="--augment"
else
AUG=""
fi
python ncc/compile.py -i data/${DS}_filtered -o data/${DS}_ncc -s data/datasets/split_${DS}.pkl -d ${DS} -c clang++-3.7.1 ${AUG}

# Preprocess datasets for MISIM models, Aroma, code2vec, and NCC
python preprocess/gnn_preprocess.py -i data/${DS}_cass -o data/datasets/${DS}/dataset-gnn -s data/datasets/split_${DS}.pkl
python preprocess/sbt_preprocess.py -i data/${DS}_cass -o data/datasets/${DS}/dataset-sbt -s data/datasets/split_${DS}.pkl
python preprocess/bof_preprocess.py -i data/${DS}_cass -o data/datasets/${DS}/dataset-bof -s data/datasets/split_${DS}.pkl
python preprocess/aroma_preprocess.py -i data/${DS}_cass -o data/datasets/${DS}/dataset-aroma -s data/datasets/split_${DS}.pkl
python code2vec/preprocess.py -i data/${DS}_c2v -o data/datasets/${DS}/dataset-c2v -s data/datasets/split_${DS}.pkl
python ncc/preprocess.py -i data/${DS}_ncc -o data/datasets/${DS}/dataset-ncc -s data/datasets/split_${DS}.pkl

done
