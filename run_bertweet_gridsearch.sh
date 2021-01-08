#!/bin/bash

#$ -wd /export/b16/justin/minerva/JHU-CUT/logs
#$ -N protest-filtration-model
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jsech1@jhu.edu
#$ -l ram_free=5G,mem_free=5G,gpu=1,hostname=b1[123456789]|c0*|c1[12345789]
#$ -q g.q
#$ -t 1 ##-42

# Activate dev environments and call programs
export LD_LIBRARY_PATH=:/opt/NVIDIA/cuda-10/lib64/
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
source /home/jsech1/miniconda3/bin/activate minerva-proj

# Define input directory, temp directory, and path to final feature output file
WORKING_DIR="${MINERVA_HOME}/JHU-CUT/"
INPUT_FILE="${WORKING_DIR}/private/labelled_text_is_general_unrest.csv"

BATCH_SIZE=(10 20 50 100 150 200)
LEARNING_RATE=(0.0001 0.001 0.01 0.05 0.1 0.2 0.5)

TASK=$[SGE_TASK_ID - 1]
#CUR_BATCH=${BATCH_SIZE[${TASK} / 7]}
#CUR_LR=${LEARNING_RATE[${TASK} % 7]}
CUR_BATCH=20
CUR_LR=0.1
# Cross-Validate Results
echo "Running model with batch_size=${CUR_BATCH} and LR=${CUR_LR}"
python "${WORKING_DIR}/bertweet_model.py" \
      --input-file "${INPUT_FILE}" \
      --results-file "${WORKING_DIR}/unrest_results/all_data_results_bs${CUR_BATCH}_lr${CUR_LR}.pkl" \
      --cross-validate \
      --batch-size "${CUR_BATCH}" \
      --learning-rate "${CUR_LR}" \
      --seed 42 \

# Check exit status
status=$?
if [ $status -ne 0 ]
then
    echo "Program failed"
    exit 1
fi
