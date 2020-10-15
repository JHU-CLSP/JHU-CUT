#!/bin/bash

#$ -wd /home/aadelucia/files/minerva/emnlp_wnut_2020
#$ -N ngram-model
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jsech1@jhu.edu
#$ -l ram_free=5G,mem_free=5G

# Activate dev environments and call programs
export LD_LIBRARY_PATH=:/opt/NVIDIA/cuda-10/lib64/
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
source /home/jsech1/miniconda3/bin/activate minerva-proj

# Define input directory, temp directory, and path to final feature output file
WORKING_DIR="${MINERVA_HOME}/emnlp_wnut_2020"
INPUT_FILE="${WORKING_DIR}/ignore/labelled_tweets_is_protest_event.csv"

# Only keywords
python "${WORKING_DIR}/ngram_model.py" \
      --input-file "${INPUT_FILE}" \
      --results-file "${WORKING_DIR}/keyword_model_results_${JOB_ID}.pkl" \
      --save-model-path "${WORKING_DIR}/keyword_model_${JOB_ID}.pkl" \
      --keywords-file "${WORKING_DIR}/keywords_english.txt" \
      --seed 42

# Check exit status
status=$?
if [ $status -ne 0 ]
then
    echo "Program failed"
    exit 1
fi

# All unigrams
python "${WORKING_DIR}/ngram_model.py" \
      --input-file "${INPUT_FILE}" \
      --results-file "${WORKING_DIR}/unigram_model_results_${JOB_ID}.pkl" \
      --save-model-path "${WORKING_DIR}/unigram_model_${JOB_ID}.pkl" \
      --max-iter 4000 \
      --seed 42

# Check exit status
status=$?
if [ $status -ne 0 ]
then
    echo "Program failed"
    exit 1
fi

