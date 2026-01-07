#!/usr/bin/env bash
set -e

pip install -r requirements.txt
python download_model.py
unzip -o xact_finetuned_model.zip
