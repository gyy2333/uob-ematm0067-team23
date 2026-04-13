#!/bin/bash
echo "current workspace: $(pwd)"

input_dir=data/
mkdir -p $input_dir
output_dir=outputs/
mkdir -p $output_dir

# pip install bertopic
# pip install seaborn
# pip install wordcloud

python src/main.py --output-path outputs/