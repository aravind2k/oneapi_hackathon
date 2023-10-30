#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
cd openai
python app.py
make run
