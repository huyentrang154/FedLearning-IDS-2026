#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
#########################################
# Parallel Experiment Scheduler
#########################################

MAX_JOBS=2

DEVICE=cuda
DEVICE_ID=0
RADIO=1.0
EXP_NAME=$(date +"%Y%m%d-%H%M%S")

mkdir -p logs

#########################################
# Datasets
#########################################

DATASETS=(
    unsw
    cic_ids
    toniot
)

#########################################
# Algorithms
#########################################

ALGOS=(
    fednoro
    fedcorr
)

#########################################
# Seeds
#########################################

SEEDS=(
0
1
)

#########################################
# Fixed parameters
#########################################

POISON_WORKERS=(
    0.3
)
#########################################

run_job(){

    DATASET=$1
    ALGO=$2
    SEED=$3
    PW=$4
    echo "================================================="
    echo "Dataset : $DATASET"
    echo "Algorithm : $ALGO"
    echo "Seed : $SEED"
    echo "POISON_WORKERS : $PW"
    echo "================================================="

    python3 main.py \
        --algorithm $ALGO \
        --dataset $DATASET \
        --device $DEVICE \
        --device_id $DEVICE_ID \
        --seed $SEED \
        --num_of_poisoned_workers $PW \
        --poisoned_sample_ratio $RADIO \
        --exp_name ${EXP_NAME} \
        > logs/${DATASET}_${ALGO}_seed${SEED}_poisoned_workers${PW}.log 2>&1

    echo "Finished : ${DATASET} ${ALGO} Seed ${SEED}"

}

#########################################

for DATASET in "${DATASETS[@]}"
do
    for ALGO in "${ALGOS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
          for POISON_WORKERS in "${POISON_WORKERS[@]}"
            do

            while [ $(jobs -pr | wc -l) -ge $MAX_JOBS ]
            do
                sleep 3
            done

            run_job "$DATASET" "$ALGO" "$SEED" "$POISON_WORKERS" &
            done
        done
    done
done

wait

echo
echo "==========================================="
echo "All experiments finished."
echo "==========================================="
