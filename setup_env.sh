#!/bin/bash

echo "Setting up Knowledge Ripple Effect experiment environment..."

conda create -n knowledge_ripple python=3.10 -y
conda activate knowledge_ripple

pip install -r requirements.txt

python -m spacy download en_core_web_sm

mkdir -p data/raw
mkdir -p data/processed
mkdir -p results/baseline
mkdir -p results/perturbation
mkdir -p results/analysis

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate knowledge_ripple"