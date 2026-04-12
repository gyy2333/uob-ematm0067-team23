#!/bin/bash
pwd
if python -c "import bertopic" >/dev/null 2>&1; then
    echo "bertopic is already installed"
else
    echo "bertopic is not installed, installing..."
    pip install bertopic
fi
python -m py_compile Modeling/modeling_bert.py Modeling/modeling_lda.py main.py
python main.py  --data-path data/modeling/ai_ml_nlp_dataset_processed.csv --output-path outputs/