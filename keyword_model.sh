#!/bin/bash

#$ -wd /export/b16/justin/minerva/emnlp_submission/
#$ -N keyword-model
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jsech1@jhu.edu
#$ -l ram_free=10G,mem_free=10G,gpu=1
##$ -q g.q

# Activate dev environments and call programs
export LD_LIBRARY_PATH=:/opt/NVIDIA/cuda-10/lib64/
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
conda activate minerva-proj

# Define input directory, temp directory, and path to final feature output file
INPUT_FILE=$MINERVA_HOME/labelled_tweets_updated.csv

python $MINERVA_HOME/emnlp_submission/keyword_model.py \
    --input-file $INPUT_FILE \
    --output-file $MINERVA_HOME/emnlp_submission/keyword_model_$JOB_ID.pkl \
    --keywords-file $MINERVA_HOME/emnlp_submission/keywords_english.txt \
    --learning-rate 0.1 \
    --seed 42

# Check exit status
status=$?
if [ $status -ne 0 ]
then
    echo "Program failed"
    exit 1
fi

