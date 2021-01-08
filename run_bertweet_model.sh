#!/bin/bash

#$ -cwd
#$ -N bertweet-filtration-model
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -l ram_free=5G,mem_free=5G,gpu=1,hostname=b1[123456789]|c0*|c1[12345789]
#$ -q g.q

# Activate dev environments and call programs
export LD_LIBRARY_PATH=:/opt/NVIDIA/cuda-10/lib64/
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
conda activate twitter-dev

# Define input directory, temp directory, and path to final feature output file
WORKING_DIR="${MINERVA_HOME}/emnlp_wnut_2020/"
INPUT_FILE="${JUSTIN}/minerva/JHU-CUT/private/labelled_text_is_general_unrest.csv"

# Run cross-validation
python "${WORKING_DIR}/bertweet_model.py" \
      --input-file "${INPUT_FILE}" \
      --cross-validate \
      --results-file "${WORKING_DIR}/results/bertweet_model_results.pkl" \
      --batch-size 20 \
      --learning-rate 0.1 \
      --seed 42 \

# Check exit status
status=$?
if [ $status -ne 0 ]
then
    echo "Cross validation failed"
    exit 1
fi

# Run final model
python "${WORKING_DIR}/bertweet_model.py" \
      --input-file "${INPUT_FILE}" \
      --save-model-path "${WORKING_DIR}/results/bertweet_model.pt" \
      --batch-size 20 \
      --learning-rate 0.1 \
      --seed 42 \

# Check exit status
status=$?
if [ $status -ne 0 ]
then
    echo "Final train failed"
    exit 1
fi
