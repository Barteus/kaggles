#!/bin/bash

pip install kaggle
rm -R data
mkdir data
kaggle competitions download -c dog-breed-identification -p data

unzip -q data/train -d data
unzip -q data/test -d data
unzip -q data/labels.csv.zip -d data
unzip -q data/sample_submission.csv.zip -d data
rm data/*.zip
