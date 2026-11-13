#!/bin/bash

<<<<<<< HEAD
# Di chuyển vào thư mục chứa file script (system/) để gọi main.py chính xác
=======
# Di chuyển vào thư mục chứa file run.sh (system/) để gọi main.py chính xác
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#########################################
# 1. Tự động dò tìm Python có cài sẵn PyTorch (hoặc cài đặt mới)
#########################################
PYTHON_BIN=""

<<<<<<< HEAD
=======
# Danh sách các đường dẫn Python tiềm năng (ưu tiên .venv, Conda envs, và Python cài trên máy)
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
CANDIDATES=(
    "../.venv/Scripts/python.exe"
    ".venv/Scripts/python.exe"
    "../.venv/bin/python"
    ".venv/bin/python"
    "python"
    "python3"
)

<<<<<<< HEAD
=======
# Bổ sung tự động các môi trường Conda và Python chuẩn trên Windows/Linux
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
for p in \
    "$HOME"/anaconda3/envs/*/python.exe \
    "$HOME"/miniconda3/envs/*/python.exe \
    "$HOME"/AppData/Local/anaconda3/envs/*/python.exe \
    "$HOME"/AppData/Local/miniconda3/envs/*/python.exe \
    /c/ProgramData/anaconda3/envs/*/python.exe \
    /c/ProgramData/miniconda3/envs/*/python.exe \
    "$HOME"/anaconda3/python.exe \
    "$HOME"/miniconda3/python.exe \
    "$HOME"/AppData/Local/anaconda3/python.exe \
    "$HOME"/AppData/Local/miniconda3/python.exe \
    /c/ProgramData/anaconda3/python.exe \
    /c/ProgramData/miniconda3/python.exe \
    "$HOME"/AppData/Local/Programs/Python/Python*/python.exe \
    /c/Python*/python.exe \
    "$HOME"/anaconda3/envs/*/bin/python \
    "$HOME"/miniconda3/envs/*/bin/python \
    "$HOME"/anaconda3/bin/python \
    "$HOME"/miniconda3/bin/python; do
    if [ -f "$p" ]; then
        CANDIDATES+=("$p")
    fi
done

echo "Đang kiểm tra môi trường Python..."

<<<<<<< HEAD
=======
# Bước 1: Ưu tiên tìm Python ĐÃ CÓ SẴN thư viện torch
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
for candidate in "${CANDIDATES[@]}"; do
    if "$candidate" -c "import torch" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        echo "-> Đã tìm thấy môi trường có sẵn PyTorch: $PYTHON_BIN"
        break
    fi
done

<<<<<<< HEAD
=======
# Bước 2: Nếu chưa môi trường nào có torch, chọn Python hợp lệ (bỏ qua MSYS/GitBash python không cài được wheel) để cài đặt
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
if [ -z "$PYTHON_BIN" ]; then
    for candidate in "${CANDIDATES[@]}"; do
        if "$candidate" -c "import sys; sys.exit(0 if 'msys' not in sys.platform.lower() else 1)" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Không tìm thấy Python hợp lệ! Hãy cài đặt Python (3.9 - 3.12) hoặc Anaconda/Miniconda trước."
    exit 1
fi

echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

<<<<<<< HEAD
=======
# Kiểm tra và cài đặt các thư viện cần thiết nếu chưa có
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
if ! "$PYTHON_BIN" -c "import torch, numpy, scipy, sklearn, pandas, matplotlib, h5py" >/dev/null 2>&1; then
    echo "================================================="
    echo "Môi trường chưa đủ thư viện. Đang cài đặt từ requirements.txt ..."
    echo "================================================="
    "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio
    "$PYTHON_BIN" -m pip install -r requirements.txt

<<<<<<< HEAD
    if ! "$PYTHON_BIN" -c "import torch, numpy, scipy, sklearn, pandas, matplotlib, h5py" >/dev/null 2>&1; then
        echo "[ERROR] Cài đặt thư viện thất bại trên $PYTHON_BIN!"
=======
    # Kiểm tra lại sau khi cài đặt
    if ! "$PYTHON_BIN" -c "import torch, numpy, scipy, sklearn, pandas, matplotlib, h5py" >/dev/null 2>&1; then
        echo "[ERROR] Cài đặt thư viện thất bại trên $PYTHON_BIN!"
        echo "Hãy chạy lệnh sau trên terminal của bạn để kiểm tra lỗi chi tiết:"
        echo "    \"$PYTHON_BIN\" -m pip install torch torchvision torchaudio numpy scipy scikit-learn pandas matplotlib h5py tqdm"
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
        exit 1
    fi
fi

#########################################
# 2. Cấu hình chung (Parallel Scheduler)
#########################################
MAX_JOBS=4

<<<<<<< HEAD
=======
# Tự động kiểm tra GPU CUDA, nếu không có CUDA sẽ chuyển sang cpu
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
if "$PYTHON_BIN" -c "import torch; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi
DEVICE_ID="0"
EXP_NAME=$(date +"%Y%m%d-%H%M%S")

mkdir -p logs

#########################################
# 3. Datasets & Data Type
#########################################
DATASETS=(
    "toniot"
)

<<<<<<< HEAD
=======
# "noniid" (tự động map sang noniid20) hoặc "exdir" (tự động map sang noniidexdir20)
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
DATA_TYPE="noniid20"

#########################################
# 4. 8 Thuật toán FL
#########################################
ALGOS=(
<<<<<<< HEAD
    "fedavg"
    "fedrep"
    "fedgh"
    "fedcorr"
    "fedhome"
    "lg_fedavg"
    "fedrep"
=======
    "fedus"
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
)

#########################################
# 5. Seeds (có thể thêm nhiều seed: (0 1 2))
#########################################
SEEDS=(
<<<<<<< HEAD
    0 1 2
)

#########################################
# 6. Kịch bản Flipping (num_of_poisoned_workers = 0.3, poisoned_sample_ratio = 0.4)
#########################################
SCENARIOS=(
    "label_flipping|0.3|1.0"
=======
    2
)

#########################################
# 6. Các kịch bản chạy với noniid (Định dạng: "NOISE_TYPE|NPW|PSR")
# Mặc định bật kịch bản Clean (noniid).
# Bỏ comment các dòng bên dưới nếu muốn chạy luôn các kịch bản Poisoning trên noniid.
#########################################
SCENARIOS=(
    "None|0.0|0.0"                  # Kịch bản 1: Clean với noniid (noniid20)
    # "label_flipping|0.3|1.0"      # Kịch bản 2a: Label flipping (npw=0.3, psr=1.0)
    # "label_flipping|0.5|1.0"      # Kịch bản 2b: Label flipping (npw=0.5, psr=1.0)
    # "Asymmetric|0.3|0.4"          # Kịch bản 3a: Asymmetric (npw=0.3, psr=0.4)
    # "Symmetric|0.3|0.4"           # Kịch bản 3b: Symmetric (npw=0.3, psr=0.4)
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
)

#########################################
# 7. Hàm thực thi từng tiến trình
#########################################
run_job() {
    local DATASET=$1
    local ALGO=$2
    local SEED=$3
    local NOISE_TYPE=$4
    local NPW=$5
    local PSR=$6

    local SCENARIO_TAG="${DATA_TYPE}_${NOISE_TYPE}_npw${NPW}_psr${PSR}"
<<<<<<< HEAD
=======
    if [ "$NOISE_TYPE" = "None" ]; then
        SCENARIO_TAG="${DATA_TYPE}_clean"
    fi

>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
    local LOG_FILE="logs/${EXP_NAME}_${DATASET}_${ALGO}_${SCENARIO_TAG}_seed${SEED}.log"

    echo "================================================="
    echo "[START] Dataset    : $DATASET ($DATA_TYPE)"
    echo "        Algorithm  : $ALGO"
    echo "        Scenario   : $SCENARIO_TAG"
    echo "        Seed       : $SEED | Device: $DEVICE"
    echo "        ConsoleLog : $LOG_FILE"
    echo "================================================="

    "$PYTHON_BIN" main.py \
        --dataset "$DATASET" \
        --data_type "$DATA_TYPE" \
        --algorithm "$ALGO" \
        --device "$DEVICE" \
        --device_id "$DEVICE_ID" \
        --seed "$SEED" \
        --noise_type "$NOISE_TYPE" \
        --num_of_poisoned_workers "$NPW" \
        --poisoned_sample_ratio "$PSR" \
        --exp_name "$EXP_NAME" \
        > "$LOG_FILE" 2>&1

    echo "[DONE]  ${DATASET} | ${ALGO} | ${SCENARIO_TAG} | Seed ${SEED}"
}

#########################################
# 8. Vòng lặp điều phối công việc
#########################################
for DATASET in "${DATASETS[@]}"; do
    for SCENARIO in "${SCENARIOS[@]}"; do
        IFS='|' read -r NOISE_TYPE NPW PSR <<< "$SCENARIO"
        for ALGO in "${ALGOS[@]}"; do
            for SEED in "${SEEDS[@]}"; do

                while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
                    sleep 3
                done

                run_job "$DATASET" "$ALGO" "$SEED" "$NOISE_TYPE" "$NPW" "$PSR" &

            done
        done
    done
done

wait

echo
echo "==========================================="
<<<<<<< HEAD
echo "All Flipping experiments finished! (EXP_NAME: $EXP_NAME)"
echo "==========================================="

=======
echo "All experiments finished! (EXP_NAME: $EXP_NAME)"
echo "==========================================="
>>>>>>> 15d875d0e935e4c4ed34fc42278501cc26b08df7
