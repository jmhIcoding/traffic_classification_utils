#!/bin/bash

source ~/venv/bin/activate && 
export PYTHONPATH=/home3/jmh/traffic_classification_utils/:$PYTHONPATH &&
python3 $1 #foolbox_example.py
