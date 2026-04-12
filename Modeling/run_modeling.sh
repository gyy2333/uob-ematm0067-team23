#!/bin/bash
pwd
pip install bertopic
python -m py_compile Modeling/modeling_bert.py Modeling/modeling_lda.py main.py