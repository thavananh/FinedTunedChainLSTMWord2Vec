#!/bin/bash
conda create -n py311 python=3.11 -y
conda init
conda activate py311
conda install -y -c conda-forge anaconda
pip install tensorflow[and-cuda] keras-tuner pyvi underthesea loky python-telegram-bot gensim matplotlib pandas

