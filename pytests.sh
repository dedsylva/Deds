#! /usr/bin/bash
rm -rf tests/__pycache__
python3 -m pytest -v tests/*

