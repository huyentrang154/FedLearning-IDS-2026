#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
#########################################
# Parallel Experiment Scheduler
#########################################

MAX_JOBS=4

DEVICE=cuda
DEVICE_ID=0



mkdir -p logs

#########################################
# Datasets
#########################################

DATASETS=(
    toniot
)

#########################################
# Algorithms
#########################################

ALGOS=(
    fedavg
    fedgh
    fedrep
    fedhome
    lg_fedavg
    fedus
)

#########################################
# Seeds
#########################################

SEEDS=(
0
1
2
)

#########################################
# Fixed parameters
#########################################

POISON_WORKERS=(
0
0.3
0.5
)

#########################################

run_job(){

    DATASET=$1
    ALGO=$2
    SEED=$3
    PW=$4
    EXP_NAME="$(date +"%Y%m%d-%H%M%S")_pw${PW}"
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
        --exp_name ${EXP_NAME} \
        > logs/${DATASET}_${ALGO}_seed${SEED}_pw${PW}.log 2>&1

    echo "Finished : ${DATASET} ${ALGO} Seed ${SEED} PW${PW}"

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