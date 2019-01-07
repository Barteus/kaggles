#!/bin/bash

ENV_NAME="$1"
OUTPUT_FILE="$ENV_NAME/environment.yml"
conda env export -n $ENV_NAME > $OUTPUT_FILE
echo "Environment $ENV_NAME exported to $OUTPUT_FILE"
