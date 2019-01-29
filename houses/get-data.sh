#!/bin/bash

pip install kaggle
rm -R data
mkdir data
kaggle competitions download -c house-prices-advanced-regression-techniques -p data

#Not needed for this competition dataset
#unzip -q data/train -d data
#unzip -q data/test -d data
#unzip -q data/labels.csv.zip -d data
#unzip -q data/sample_submission.csv.zip -d data
#rm data/*.zip
