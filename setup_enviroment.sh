#!/bin/bash
conda create -n py311 python=3.11 -y
conda init
conda activate py311
conda install -y -c conda-forge anaconda
conda install -y -c conda-forge tensorflow[and-cuda] keras-tuner pyvi underthesea loky
