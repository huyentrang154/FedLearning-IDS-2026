#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#########################################
# Parallel Experiment Scheduler
#########################################

MAX_JOBS=5

DEVICE=cuda
DEVICE_ID=0
EXP_NAME=$(date +"%Y%m%d-%H%M%S")

# Clean Non-IID experiment
NOISYTYPE=None
POISON_WORKERS=0
POISON_RATIO=0
DATA_TYPE=noniid

mkdir -p logs

#########################################
# Datasets
#########################################

DATASETS=(
    nslkdd
    unsw
    cic_ids
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
    fednoro
    fedcorr
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

run_job(){

    DATASET=$1
    ALGO=$2
    SEED=$3

    echo "================================================="
    echo "Dataset : $DATASET"
    echo "Algorithm : $ALGO"
    echo "Seed : $SEED"
    echo "Data type : $DATA_TYPE"
    echo "Noise type : $NOISYTYPE"
    echo "================================================="

    python3 main.py \
        --algorithm $ALGO \
        --dataset $DATASET \
        --device $DEVICE \
        --device_id $DEVICE_ID \
        --seed $SEED \
        --data_type $DATA_TYPE \
        --noise_type $NOISYTYPE \
        --num_of_poisoned_workers $POISON_WORKERS \
        --poisoned_sample_ratio $POISON_RATIO \
        --exp_name ${EXP_NAME} \
        > logs/${DATASET}_${ALGO}_seed${SEED}.log 2>&1

    echo "Finished : ${DATASET} ${ALGO} Seed ${SEED}"
}

#########################################

for DATASET in "${DATASETS[@]}"
do
    for ALGO in "${ALGOS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do

            while [ $(jobs -pr | wc -l) -ge $MAX_JOBS ]
            do
                sleep 3
            done

            run_job "$DATASET" "$ALGO" "$SEED" &

        done
    done
done

wait

echo
echo "==========================================="
echo "All experiments finished."
echo "==========================================="#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#########################################
# Parallel Experiment Scheduler
#########################################

MAX_JOBS=3

DEVICE=cuda
DEVICE_ID=0
EXP_NAME=$(date +"%Y%m%d-%H%M%S")

# Clean Non-IID experiment
NOISYTYPE=None
POISON_WORKERS=0
POISON_RATIO=0
DATA_TYPE=noniid

mkdir -p logs

#########################################
# Datasets
#########################################

DATASETS=(
    nslkdd
    unsw
    cic_ids
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
    3
    4
)

#########################################

run_job(){

    DATASET=$1
    ALGO=$2
    SEED=$3

    echo "================================================="
    echo "Dataset : $DATASET"
    echo "Algorithm : $ALGO"
    echo "Seed : $SEED"
    echo "Data type : $DATA_TYPE"
    echo "Noise type : $NOISYTYPE"
    echo "================================================="

    python3 main.py \
        --algorithm $ALGO \
        --dataset $DATASET \
        --device $DEVICE \
        --device_id $DEVICE_ID \
        --seed $SEED \
        --data_type $DATA_TYPE \
        --noise_type $NOISYTYPE \
        --num_of_poisoned_workers $POISON_WORKERS \
        --poisoned_sample_ratio $POISON_RATIO \
        --exp_name ${EXP_NAME} \
        > logs/${DATASET}_${ALGO}_seed${SEED}.log 2>&1

    echo "Finished : ${DATASET} ${ALGO} Seed ${SEED}"
}

#########################################

for DATASET in "${DATASETS[@]}"
do
    for ALGO in "${ALGOS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do

            while [ $(jobs -pr | wc -l) -ge $MAX_JOBS ]
            do
                sleep 3
            done

            run_job "$DATASET" "$ALGO" "$SEED" &

        done
    done
done

wait

echo
echo "==========================================="
echo "All experiments finished."
echo "==========================================="#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#########################################
# Parallel Experiment Scheduler
#########################################

MAX_JOBS=3

DEVICE=cuda
DEVICE_ID=0
EXP_NAME=$(date +"%Y%m%d-%H%M%S")

# Clean Non-IID experiment
NOISYTYPE=None
POISON_WORKERS=0
POISON_RATIO=0
DATA_TYPE=noniid

mkdir -p logs

#########################################
# Datasets
#########################################

DATASETS=(
    nslkdd
    unsw
    cic_ids
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
    3
    4
)

#########################################

run_job(){

    DATASET=$1
    ALGO=$2
    SEED=$3

    echo "================================================="
    echo "Dataset : $DATASET"
    echo "Algorithm : $ALGO"
    echo "Seed : $SEED"
    echo "Data type : $DATA_TYPE"
    echo "Noise type : $NOISYTYPE"
    echo "================================================="

    python3 main.py \
        --algorithm $ALGO \
        --dataset $DATASET \
        --device $DEVICE \
        --device_id $DEVICE_ID \
        --seed $SEED \
        --data_type $DATA_TYPE \
        --noise_type $NOISYTYPE \
        --num_of_poisoned_workers $POISON_WORKERS \
        --poisoned_sample_ratio $POISON_RATIO \
        --exp_name ${EXP_NAME} \
        > logs/${DATASET}_${ALGO}_seed${SEED}.log 2>&1

    echo "Finished : ${DATASET} ${ALGO} Seed ${SEED}"
}

#########################################

for DATASET in "${DATASETS[@]}"
do
    for ALGO in "${ALGOS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do

            while [ $(jobs -pr | wc -l) -ge $MAX_JOBS ]
            do
                sleep 3
            done

            run_job "$DATASET" "$ALGO" "$SEED" &

        done
    done
done

wait

echo
echo "==========================================="
echo "All experiments finished."
echo "==========================================="#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#########################################
# Parallel Experiment Scheduler
#########################################

MAX_JOBS=3

DEVICE=cuda
DEVICE_ID=0
EXP_NAME=$(date +"%Y%m%d-%H%M%S")

# Clean Non-IID experiment
NOISYTYPE=None
POISON_WORKERS=0
POISON_RATIO=0
DATA_TYPE=noniid

mkdir -p logs

#########################################
# Datasets
#########################################

DATASETS=(
    nslkdd
    unsw
    cic_ids
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
    3
    4
)

#########################################

run_job(){

    DATASET=$1
    ALGO=$2
    SEED=$3

    echo "================================================="
    echo "Dataset : $DATASET"
    echo "Algorithm : $ALGO"
    echo "Seed : $SEED"
    echo "Data type : $DATA_TYPE"
    echo "Noise type : $NOISYTYPE"
    echo "================================================="

    python3 main.py \
        --algorithm $ALGO \
        --dataset $DATASET \
        --device $DEVICE \
        --device_id $DEVICE_ID \
        --seed $SEED \
        --data_type $DATA_TYPE \
        --noise_type $NOISYTYPE \
        --num_of_poisoned_workers $POISON_WORKERS \
        --poisoned_sample_ratio $POISON_RATIO \
        --exp_name ${EXP_NAME} \
        > logs/${DATASET}_${ALGO}_seed${SEED}.log 2>&1

    echo "Finished : ${DATASET} ${ALGO} Seed ${SEED}"
}

#########################################

for DATASET in "${DATASETS[@]}"
do
    for ALGO in "${ALGOS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do

            while [ $(jobs -pr | wc -l) -ge $MAX_JOBS ]
            do
                sleep 3
            done

            run_job "$DATASET" "$ALGO" "$SEED" &

        done
    done
done

wait

echo
echo "==========================================="
echo "All experiments finished."
echo "==========================================="