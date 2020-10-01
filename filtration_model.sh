#!/bin/bash

#$ -wd /export/b16/justin/minerva/emnlp_submission/
#$ -N filtration-model
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jsech1@jhu.edu
#$ -l ram_free=5G,mem_free=5G,gpu=1,hostname=b1[123456789]|c0*|c1[12345789]
#$ -q g.q

# Activate dev environments and call programs
export LD_LIBRARY_PATH=:/opt/NVIDIA/cuda-10/lib64/
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
source /home/jsech1/miniconda3/bin/activate minerva-proj

# Define input directory, temp directory, and path to final feature output file
INPUT_FILE=$MINERVA_HOME/emnlp_submission/labelled_tweets_updated.csv

python $MINERVA_HOME/emnlp_submission/filtration_model.py \
      --input-file $INPUT_FILE \
      --output-file $MINERVA_HOME/emnlp_submission/filtration_model_$JOB_ID.pkl \
      --cross-validate \
      --save-preds \
      --batch-size 20 \
      --learning-rate 0.5 \
      --seed 42

# Check exit status
status=$?
if [ $status -ne 0 ]
then
    echo "Program failed"
    exit 1
fi

