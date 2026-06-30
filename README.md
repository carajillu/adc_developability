# adc_developability
Workflows used in the paper "Something Something Developability of Antibody-Drug Conjugates"

# Installation

Beware, this is a "works on my machine" installation guide, writing it for myself so that I know what I've done.

## Cloning the repo
```bash
cd $GENERIC_SOFTWARE_FOLDER # this is ~/github for me
git clone https://github.com/carajillu/adc_developability.git
cd adc_developability
```

## Environment
It is recommended to create a .venv or .conda environment to run these workflows:
### .venv
```bash
python -m venv .venv
source .venv/bin/activate # version will be whichever python version you have installed
```
### .conda
```bash
conda create --prefix $(pwd)/.conda python=3.14.3 # with conda you can specify version which might be handy
conda activate $(pwd)/.conda
```

## Requirements
- pyTorch (torch) installed via pyproject.toml. Version TBC
- transformers installed via pyproject.toml. Version TBC

### redim
This is the dimensionality reduction package I wrote to make things easier for myself.
To install
```bash
cd $GENERIC_SOFTWARE_FOLDER # this is ~/github for me
git clone https://gitlab.cis.strath.ac.uk/ixb26106/redim.git
cd redim
pip install . # dependency versions should be pretty standard
```

## Installation
```bash
cd $GENERIC_SOFTWARE_FOLDER # this is ~/github for me
cd adc_developability
pip install .
```

# Running the pipeline
I will first write the pipeline, then explain here how to run it

