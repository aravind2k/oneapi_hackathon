#!/bin/bash
conda init powershell
conda activate pytorch
cd openai/
pip install -r requirements.txt
streamlit run streamlit.py
