#!/bin/bash

pip install kaggle
rm -R data
mkdir data
kaggle competitions download -c humpback-whale-identification -p data

unzip -q data/train -d data/train
unzip -q data/test -d data/test
rm data/*.zip
