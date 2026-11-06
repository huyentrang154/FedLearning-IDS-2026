# FedNoRo và FedCorr — Hướng dẫn tham số và chạy thực nghiệm

> **Môi trường:** `source ~/venv/fl/bin/activate`
> **Thư mục chạy:** thư mục chứa `main.py`

---

## Mục lục

1. [Tổng quan thuật toán](#1-tổng-quan-thuật-toán)
2. [Datasets hỗ trợ](#2-datasets-hỗ-trợ)
3. [Tham số chung](#3-tham-số-chung)
4. [FedNoRo — Tham số và lệnh chạy](#4-fednoro--tham-số-và-lệnh-chạy)
5. [FedCorr — Tham số và lệnh chạy](#5-fedcorr--tham-số-và-lệnh-chạy)
6. [Kịch bản tấn công](#6-kịch-bản-tấn-công)
7. [Lưu ý quan trọng](#7-lưu-ý-quan-trọng)

---

## 1. Tổng quan thuật toán

### FedNoRo
**Paper:** *FedNoRo: Towards Noise-Robust Federated Learning by Addressing Class Imbalance and Label Noise Heterogeneity* (IJCAI 2023)

Hai giai đoạn:
- **Stage 1 — Warm-up** (`rounds 0 … warmup_rounds-1`): toàn bộ clients huấn luyện BCE bình thường, server tổng hợp FedAvg.
- **Stage 2 — Robust** (`rounds warmup_rounds … global_rounds`):
  - Server dùng ensemble 9 GMM trên vector tổn thất per-class để nhận diện noisy clients một lần duy nhất.
  - **Clean clients:** vẫn dùng BCE thường.
  - **Noisy clients:** dùng Knowledge Distillation — pha trộn soft-BCE từ teacher (global model) với hard-BCE từ nhãn cục bộ. Trọng số `w_kd` được tăng dần theo lịch sigmoid.
  - Server tổng hợp bằng **DaAgg**: down-weight noisy clients dựa theo khoảng cách model đến clean client gần nhất.

**Đặc điểm trong codebase này:** Sử dụng phân loại nhị phân (BCE + sigmoid, `num_classes=1`) thay vì multi-class như bản gốc, phù hợp với bài toán phát hiện xâm nhập.

---

### FedCorr
**Paper:** *FedCorr: Multi-Stage Federated Learning for Label Noise Correction* (CVPR 2022)

Ba giai đoạn:
- **Stage 1 — Noise Detection** (`iteration1` vòng lặp ngoài):
  - Mỗi vòng: huấn luyện cục bộ với proximal term `beta * mu * ||w - w_g||`, rồi tính LID (Local Intrinsic Dimensionality) trên softmax output.
  - Tích lũy LID qua các vòng → GMM phân loại clean/noisy client.
  - GMM thứ hai trên per-sample loss → sửa nhãn sai (nếu bật `--correction 1`).
- **Stage 2 — Fine-tuning** (`rounds1` rounds, nếu bật `--fine_tuning 1`): chỉ clean clients tham gia.
- **Stage 3 — Final FedAvg** (`rounds2` rounds): toàn bộ clients.

**Đặc điểm trong codebase này:** Sử dụng multi-class (CrossEntropy, `num_classes=K`) như bản gốc.

---

## 2. Datasets hỗ trợ

| Dataset | `--dataset` | Features | Classes | Thư mục dữ liệu |
|---|---|---|---|---|
| NSL-KDD | `nslkdd` | 122 | 5 | `dataset/nslkdd/noniid20/` |
| UNSW-NB15 | `unsw` | 196 | 10 | `dataset/unsw/noniid20/` |
| CIC-IDS | `cic_ids` | 80 | 12 | `dataset/cic_ids/noniid20/` |

---

## 3. Tham số chung

Các tham số này áp dụng cho cả hai phương pháp:

| Tham số | Flag | Default | Ghi chú |
|---|---|---|---|
| Thuật toán | `-algo` | `fedus` | Dùng `fednoro` hoặc `fedcorr` |
| Dataset | `-data` | `cic_ids` | Xem bảng trên |
| Thiết bị | `-dev` | `cuda` | `cpu` nếu không có GPU |
| Số clients | `-nc` | `20` | Phải khớp với thư mục dữ liệu |
| Batch size | `-lbs` | `4` | FedNoRo paper: 16; FedCorr paper: 10 |
| Learning rate | `-lr` | `0.01` | FedNoRo paper: 3e-4; FedCorr paper: 0.03 |
| Local steps/epoch | `-ls` | `5` | Khớp cả hai paper |
| Tổng rounds | `-gr` | `50` | FedNoRo paper: 100; FedCorr paper: tuỳ stage |
| Join ratio | `-jr` | `1.0` | Tỉ lệ clients tham gia mỗi round |
| Seed | `--seed` | `0` | Để tái lập kết quả |
| Data partition | `-dt` | `noniid` | Loại phân chia dữ liệu |

---

## 4. FedNoRo — Tham số và lệnh chạy

### 4.1 Tham số đặc thù

| Tham số | Flag | Default | Ghi chú |
|---|---|---|---|
| Warm-up rounds | `--warmup_rounds` | `10` | Khớp paper: `--s1 10` |
| KD rampup bắt đầu | `--kd_begin` | `10` | Khớp paper: `--begin 10` |
| KD rampup kết thúc | `--kd_end` | `49` | Phải = `global_rounds - 1` |
| KD weight tối đa | `--kd_a` | `0.8` | Khớp paper: `--a 0.8` |
| Loại partition (log) | `--patition` | `noniid` | Chỉ dùng cho đường dẫn kết quả |
| Alpha Dirichlet (log) | `--alpha_dir` | `0.5` | Chỉ dùng cho đường dẫn kết quả |
| Kiểu tấn công (log) | `--poison_attack` | `None` | Chỉ dùng cho đường dẫn kết quả |

> **Lưu ý `kd_end`:** Giá trị mặc định `49` được thiết kế cho `global_rounds=50`.
> Nếu thay đổi `--global_rounds N`, phải thêm `--kd_end $((N-1))`.

### 4.2 Kịch bản không có tấn công (no-noise / no-attack)

```bash
# NSL-KDD
python main.py \
    -algo fednoro \
    -data nslkdd \
    -dev cpu \
    -nc 20 \
    -lbs 16 \
    -lr 0.0003 \
    -gr 50 \
    -ls 5 \
    --warmup_rounds 10 \
    --kd_begin 10 \
    --kd_end 49 \
    --kd_a 0.8 \
    --seed 0

# UNSW-NB15
python main.py \
    -algo fednoro \
    -data unsw \
    -dev cpu \
    -nc 20 \
    -lbs 16 \
    -lr 0.0003 \
    -gr 50 \
    -ls 5 \
    --warmup_rounds 10 \
    --kd_begin 10 \
    --kd_end 49 \
    --kd_a 0.8 \
    --seed 0

# CIC-IDS
python main.py \
    -algo fednoro \
    -data cic_ids \
    -dev cpu \
    -nc 20 \
    -lbs 16 \
    -lr 0.0003 \
    -gr 50 \
    -ls 5 \
    --warmup_rounds 10 \
    --kd_begin 10 \
    --kd_end 49 \
    --kd_a 0.8 \
    --seed 0
```

### 4.3 Kịch bản bị tấn công (noisy labels)

Thêm các tham số tấn công sau:

| Tham số | Flag | Giá trị gợi ý | Ý nghĩa |
|---|---|---|---|
| Kiểu nhiễu | `--noise_type` | `label_flipping` | Loại tấn công nhãn |
| Tỉ lệ clients bị tấn công | `--num_of_poisoned_workers` | `0.3` | 30% clients bị nhiễu |
| Tỉ lệ mẫu bị lật nhãn | `--poisoned_sample_ratio` | `0.5` | 50% mẫu trong client bị nhiễu |

**Noise types:**
- `label_flipping`: lật nhãn ngẫu nhiên sang lớp khác
- `Symmetric`: lật nhãn đối xứng (mỗi lớp → lớp khác theo phân phối đều)
- `Asymmetric`: lật nhãn bất đối xứng (lớp i → lớp i+1 theo vòng)

```bash
# NSL-KDD — Label Flipping, 30% clients bị tấn công, 50% mẫu nhiễu
python main.py \
    -algo fednoro \
    -data nslkdd \
    -dev cpu \
    -nc 20 \
    -lbs 16 \
    -lr 0.0003 \
    -gr 50 \
    -ls 5 \
    --warmup_rounds 10 \
    --kd_begin 10 \
    --kd_end 49 \
    --kd_a 0.8 \
    --noise_type label_flipping \
    --num_of_poisoned_workers 0.3 \
    --poisoned_sample_ratio 0.5 \
    --poison_attack label_flipping \
    --seed 0

# NSL-KDD — Asymmetric Noise, 50% clients bị tấn công
python main.py \
    -algo fednoro \
    -data nslkdd \
    -dev cpu \
    -nc 20 \
    -lbs 16 \
    -lr 0.0003 \
    -gr 50 \
    -ls 5 \
    --warmup_rounds 10 \
    --kd_begin 10 \
    --kd_end 49 \
    --kd_a 0.8 \
    --noise_type Asymmetric \
    --num_of_poisoned_workers 0.5 \
    --poisoned_sample_ratio 0.4 \
    --poison_attack Asymmetric \
    --seed 0
```

### 4.4 Script chạy nhiều seed (FedNoRo)

```bash
#!/bin/bash
source ~/venv/fl/bin/activate

DATASET=nslkdd
NOISE=label_flipping
POISON_WORKERS=0.3
PSR=0.5

for SEED in 0 1 2 3; do
    python main.py \
        -algo fednoro \
        -data $DATASET \
        -dev cpu \
        -nc 20 \
        -lbs 16 \
        -lr 0.0003 \
        -gr 50 \
        -ls 5 \
        --warmup_rounds 10 \
        --kd_begin 10 \
        --kd_end 49 \
        --kd_a 0.8 \
        --noise_type $NOISE \
        --num_of_poisoned_workers $POISON_WORKERS \
        --poisoned_sample_ratio $PSR \
        --poison_attack $NOISE \
        --seed $SEED \
        > logs/fednoro_${DATASET}_${NOISE}_seed${SEED}.log 2>&1
    echo "Done: seed=$SEED"
done
```

---

## 5. FedCorr — Tham số và lệnh chạy

### 5.1 Tham số đặc thù

| Tham số | Flag | Default | Ghi chú |
|---|---|---|---|
| Vòng lặp ngoài Stage 1 | `--iteration1` | `5` | Khớp paper |
| Fine-tuning rounds | `--rounds1` | `10` | Paper: 200; scale-down cho 20 clients |
| Final FedAvg rounds | `--rounds2` | `40` | Paper: 200; scale-down cho 20 clients |
| Client fraction Stage 1 | `--frac1` | `1.0` | Paper: 0.01 cho 100 clients; 1.0 hợp lý cho 20 clients |
| Client fraction Stage 2/3 | `--frac2` | `1.0` | Paper: 0.1 cho 100 clients |
| Proximal coefficient | `--beta_fedcorr` | `0.0` | **Mặc định tắt** — bật khi có noise: dùng `1.0` |
| Tỉ lệ mẫu relabel | `--relabel_ratio` | `0.5` | Khớp paper |
| Ngưỡng confidence | `--confidence_thres` | `0.5` | Khớp paper |
| Ngưỡng phân loại clean | `--clean_set_thres` | `0.1` | Khớp paper |
| Label correction | `--correction` | `0` | **Mặc định tắt** — bật khi có noise: dùng `1` |
| Fine-tuning Stage 2 | `--fine_tuning` | `0` | **Mặc định tắt** — bật khi có noise: dùng `1` |

> **Tổng rounds FedCorr** = `iteration1 × n_sub_rounds` (Stage 1) + `rounds1` (Stage 2, nếu bật) + `rounds2` (Stage 3).
> Với defaults: 5 × 1 + 10 + 40 = **55 rounds** tổng cộng.

### 5.2 Kịch bản không có tấn công (no-noise / no-attack)

Trong kịch bản không có nhiễu, các tính năng sửa nhãn và fine-tuning không cần thiết (giữ default = tắt):

```bash
# NSL-KDD — không có tấn công
python main.py \
    -algo fedcorr \
    -data nslkdd \
    -dev cpu \
    -nc 20 \
    -lbs 10 \
    -lr 0.03 \
    -ls 5 \
    --iteration1 5 \
    --rounds1 10 \
    --rounds2 40 \
    --frac1 1.0 \
    --frac2 1.0 \
    --beta_fedcorr 0.0 \
    --correction 0 \
    --fine_tuning 0 \
    --seed 0

# UNSW-NB15 — không có tấn công
python main.py \
    -algo fedcorr \
    -data unsw \
    -dev cpu \
    -nc 20 \
    -lbs 10 \
    -lr 0.03 \
    -ls 5 \
    --iteration1 5 \
    --rounds1 10 \
    --rounds2 40 \
    --frac1 1.0 \
    --frac2 1.0 \
    --beta_fedcorr 0.0 \
    --correction 0 \
    --fine_tuning 0 \
    --seed 0

# CIC-IDS — không có tấn công
python main.py \
    -algo fedcorr \
    -data cic_ids \
    -dev cpu \
    -nc 20 \
    -lbs 10 \
    -lr 0.03 \
    -ls 5 \
    --iteration1 5 \
    --rounds1 10 \
    --rounds2 40 \
    --frac1 1.0 \
    --frac2 1.0 \
    --beta_fedcorr 0.0 \
    --correction 0 \
    --fine_tuning 0 \
    --seed 0
```

### 5.3 Kịch bản bị tấn công (noisy labels)

Khi có tấn công, bật đầy đủ tính năng chống nhiễu của FedCorr:

```bash
# NSL-KDD — Label Flipping, 30% clients, 50% mẫu nhiễu — full FedCorr
python main.py \
    -algo fedcorr \
    -data nslkdd \
    -dev cpu \
    -nc 20 \
    -lbs 10 \
    -lr 0.03 \
    -ls 5 \
    --iteration1 5 \
    --rounds1 10 \
    --rounds2 40 \
    --frac1 1.0 \
    --frac2 1.0 \
    --beta_fedcorr 1.0 \
    --correction 1 \
    --fine_tuning 1 \
    --relabel_ratio 0.5 \
    --confidence_thres 0.5 \
    --clean_set_thres 0.1 \
    --noise_type label_flipping \
    --num_of_poisoned_workers 0.3 \
    --poisoned_sample_ratio 0.5 \
    --seed 0

# NSL-KDD — Symmetric Noise, 50% clients, 40% mẫu nhiễu
python main.py \
    -algo fedcorr \
    -data nslkdd \
    -dev cpu \
    -nc 20 \
    -lbs 10 \
    -lr 0.03 \
    -ls 5 \
    --iteration1 5 \
    --rounds1 10 \
    --rounds2 40 \
    --frac1 1.0 \
    --frac2 1.0 \
    --beta_fedcorr 1.0 \
    --correction 1 \
    --fine_tuning 1 \
    --relabel_ratio 0.5 \
    --confidence_thres 0.5 \
    --clean_set_thres 0.1 \
    --noise_type Symmetric \
    --num_of_poisoned_workers 0.5 \
    --poisoned_sample_ratio 0.4 \
    --seed 0
```

### 5.4 Script chạy nhiều seed (FedCorr)

```bash
#!/bin/bash
source ~/venv/fl/bin/activate

DATASET=nslkdd
NOISE=label_flipping
POISON_WORKERS=0.3
PSR=0.5

for SEED in 0 1 2 3; do
    python main.py \
        -algo fedcorr \
        -data $DATASET \
        -dev cpu \
        -nc 20 \
        -lbs 10 \
        -lr 0.03 \
        -ls 5 \
        --iteration1 5 \
        --rounds1 10 \
        --rounds2 40 \
        --beta_fedcorr 1.0 \
        --correction 1 \
        --fine_tuning 1 \
        --noise_type $NOISE \
        --num_of_poisoned_workers $POISON_WORKERS \
        --poisoned_sample_ratio $PSR \
        --seed $SEED \
        > logs/fedcorr_${DATASET}_${NOISE}_seed${SEED}.log 2>&1
    echo "Done: seed=$SEED"
done
```

---

## 6. Kịch bản tấn công

### Các loại tấn công

| `--noise_type` | Mô tả | Tham số liên quan |
|---|---|---|
| `label_flipping` | Lật nhãn sang lớp ngẫu nhiên khác | `--poisoned_sample_ratio` |
| `Symmetric` | Lật nhãn đối xứng (phân phối đều sang các lớp khác) | `--poisoned_sample_ratio` |
| `Asymmetric` | Lật nhãn có hướng (lớp i → lớp i+1 mod K) | `--poisoned_sample_ratio` |
| `gaussian_noise` | Thêm nhiễu Gaussian vào features | `--attack_noise_std` |

### Các mức tỉ lệ tấn công thường dùng

| Kịch bản | `--num_of_poisoned_workers` | `--poisoned_sample_ratio` |
|---|---|---|
| Nhẹ | `0.2` | `0.2` |
| Trung bình | `0.3` | `0.5` |
| Nặng | `0.5` | `0.5` |
| Cực nặng | `0.5` | `1.0` |

---

## 7. Lưu ý quan trọng

### FedNoRo

1. **Coupling `kd_end` và `global_rounds`**: Mặc định `kd_end=49` được thiết kế cho `global_rounds=50`. Nếu tăng số rounds, phải điều chỉnh:
   ```bash
   -gr 100 --kd_end 99
   ```

2. **Phân loại nhị phân**: FedNoRo trong codebase này dùng BCE + sigmoid (binary), không phải CrossEntropy + softmax như bản gốc. Đây là adaptation có chủ ý cho bài toán phát hiện xâm nhập.

3. **GMM trong kịch bản no-noise**: Khi không có nhiễu, GMM có thể nhận diện nhầm một số clients là noisy (do phân phối dữ liệu non-IID). Đây là hành vi bình thường — thuật toán vẫn chạy được, chỉ là không cần thiết phải phân biệt.

4. **Kết quả lưu tại:** `../results-binary-noniid-0.5-attack-None/nslkdd/noniid/fednoro/<timestamp>/`

### FedCorr

1. **Ba flags điều khiển behavior**: Cẩn thận phân biệt giữa kịch bản:
   - **No-noise** (baseline): `--beta_fedcorr 0.0 --correction 0 --fine_tuning 0`
   - **Full FedCorr** (khi có noise): `--beta_fedcorr 1.0 --correction 1 --fine_tuning 1`

2. **Tổng số rounds**: Với `--iteration1 5 --rounds1 10 --rounds2 40`, tổng = 5 + 10 + 40 = **55 rounds** (Stage 1 chạy 1 sub-round vì `frac1=1.0`).

3. **LID và kịch bản no-noise**: Khi model đã hội tụ tốt (no-noise), softmax outputs gần nhau → khoảng cách LID ≈ 0. GMM vẫn có thể phân loại một số clients là noisy. Đây là giới hạn của LID khi không có nhiễu thực sự.

4. **Kết quả lưu tại:** `../results/<dataset>/noniid/fedcorr/<timestamp>/`

### So sánh nhanh: nên chọn gì?

| Tiêu chí | FedNoRo | FedCorr |
|---|---|---|
| Loại bài toán | Binary (0/1) | Multi-class |
| Phát hiện noisy client | GMM trên per-class loss | GMM trên LID tích lũy |
| Sửa nhãn | Không (dùng KD) | Có (relabelling) |
| Aggregation | DaAgg (distance-aware) | FedAvg thường |
| Phù hợp khi | Biết rõ class bình thường vs tấn công | Nhiều lớp nhãn phức tạp |
