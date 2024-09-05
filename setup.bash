#!/usr/bin/env bash

# Install flash-sigmoid.
echo "Attempting uninstall + reinstall of flash sigmoid ..."
pip uninstall -y flash_sigmoid
cd flash_sigmoid

MAX_JOBS=8 python3 setup.py install  # If you face issues with random deaths turn MAX_JOBS down.
cd ..
echo "Completed Flash-Sigmoid installation [check for any errors in logs above]..."

# Install optorch
echo "Starting optorch install..."
cd optorch
pip install .
cd ..
echo "Completed optorch installation [check for any errors in logs above]..."

# Install attention_simulator
echo "Starting install of attention_simulator..."
pip install python-Levenshtein
cd attention_simulator
pip install .
cd ..
echo "Completed attention_simulator installation [check for any errors in logs above]..."
