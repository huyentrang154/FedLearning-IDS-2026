# Federated Learning for Intrusion Detection

This repository implements federated learning (FL) algorithms for intrusion detection and evaluates them under non-IID client data and label-noise attacks.

The experimental plan covers four datasets, two non-IID partitions, eight target algorithms, four attack settings, and three random seeds. The command-line options documented below were checked against the supplied `system/main.py`.

## 1. Datasets

The current `main.py` defines the following dataset identifiers and input dimensions:

| CLI identifier | Dataset | Input features | Classes |
|---|---|---:|---:|
| `nslkdd` | NSL-KDD | 122 | 5 |
| `unsw` | UNSW-NB15 | 196 | 10 |
| `cic_ids` | CIC-IDS2017 | 80 | 12 |
| `toniot` | TON_IoT | 42 | 10 |

These dimensions and class counts are hard-coded in `run(args)` in `main.py`. Confirm that the processed data files use the same feature representation and label mapping.

## 2. Data partitions

The intended experiment partitions are:

| Experiment partition | Description |
|---|---|
| `noniid20` | Pathological non-IID (PAT), 20 clients |
| `noniid20exdir` | ExDir / Dirichlet non-IID, 20 clients, concentration parameter `alpha = 0.1` |

### How the current `main.py` selects data

The supplied `main.py` exposes:

```bash
-dt DATA_TYPE
--data_type DATA_TYPE
```

Its default is `noniid`. The parser does **not** directly declare `noniid20` or `noniid20exdir` as partition choices. The data loader and the dataset-generation scripts determine the exact file naming and accepted `data_type` values. Check `system/utils/data_utils.py` and the generated dataset directories before running the full matrix.

The option `--patition` (spelled this way in the current code) is described in `main.py` as a FedNoRo result-path option. Do not treat it as the general dataset partition selector unless the server implementation explicitly uses it for that purpose.


## Results Directory Structure

The `results/` directory stores experimental logs and evaluation
results generated during federated learning experiments. It is
excluded from GitHub because experimental outputs can be large and
depend on the experimental configuration.

### Directory Organization

The directory structure is organized as follows:

```text
results/
├── <dataset>/
│   └── <data_type>/
│       └── <algorithm>/
│           └── <experiment_name>/
│               └── seed_<seed>/
│                   └── tracking.log
└── summary/
    ├── summary_per_seed.csv
    ├── summary_mean_std.csv
    └── summary_mean_std.xlsx
```

The directory components are defined as follows:

| Component | Description |
|---|---|
| `<dataset>` | Dataset used in the experiment, e.g., `nslkdd`, `unsw`, `cic_ids`, or `toniot`. |
| `<data_type>` | Data partition setting, e.g., `noniid` or another supported partition type. |
| `<algorithm>` | Federated learning algorithm, e.g., `fedavg`, `fedcorr`, or `fednoro`. |
| `<experiment_name>` | Experiment identifier, typically a timestamp such as `20261022-213705`. |
| `seed_<seed>` | Random seed used for the experiment. The current setup uses `seed_0`, `seed_1`, and `seed_2`. |
| `tracking.log` | Log containing experiment configuration, training progress, and evaluation metrics. |

### Example

```text
results/
└── unsw/
    └── noniid/
        └── fedcorr/
            └── 20261022-213705/
                ├── seed_0/
                │   └── tracking.log
                ├── seed_1/
                │   └── tracking.log
                └── seed_2/
                    └── tracking.log
```

In this example, the `fedcorr` algorithm is evaluated on the
UNSW-NB15 dataset under the `noniid` partition setting. The experiment
is repeated using three random seeds.

### Result Aggregation

The script `summarize_results.py` scans the local `results/`
directory and extracts the final `Best performance metrics` block
from each valid `tracking.log`.

The script summarizes the following metrics when available:

- Accuracy
- F1-score
- Recall
- Precision

Metrics are collected separately for `normal` and `flipping`
evaluation results when those fields are present in the log.

The script uses `seed_0`, `seed_1`, and `seed_2` to calculate the
mean and sample standard deviation across runs. It generates:

- `results/summary/summary_per_seed.csv`: detailed results for each log.
- `results/summary/summary_mean_std.csv`: aggregated mean and standard deviation.
- `results/summary/summary_mean_std.xlsx`: Excel workbook containing the aggregated results and per-log details.

Run the aggregation script from the repository root:

```bash
pip install pandas openpyxl
python summarize_results.py
```

### Version Control

The `results/` directory is not tracked by GitHub. Each user must
run the experiments locally to generate their own logs and results.
The directory structure and aggregation procedure are documented
here to support reproducibility.


## 3. Algorithms

The supplied `main.py` registers these algorithm identifiers:

| Algorithm | Identifier passed to `-algo` |
|---|---|
| FedAvg | `fedavg` |
| FedGH | `fedgh` |
| FedHome | `fedhome` |
| FedRep | `fedrep` |
| LG-FedAvg | `lg_fedavg` |
| FedPURE | `fedus` |
| FedCorr | `fedcorr` |
| FedNoRo | `fednoro` |

**FedPURE note:** The requested method name is FedPURE, but the current implementation imports `FedUS` and registers it under `fedus`. Use `-algo fedus` unless you rename/register the implementation in `main.py`.

**LG-FedAvg note:** The identifier in the supplied source is `lg_fedavg` (with an underscore), not `lgfedavg`.

The same `main.py` also registers other algorithms, including FedProx, MOON, SCAFFOLD, FedPer, FedALA, APFL, APPLE, FedFomo, FedHomeLG, FedMedian, FedTrimmedMean, FLTrust, and FedAE.

## 4. Attack scenarios

### 4.1 Label flipping

The parser exposes these options:

| Parameter | CLI option | Meaning | Requested values |
|---|---|---|---|
| Poisoned-client ratio (`rc`) | `--num_of_poisoned_workers` / `-npw` | Fraction of clients selected for poisoning | `0.3`, `0.5` |
| Poisoned-sample ratio (`rs`) | `--poisoned_sample_ratio` / `-psr` | Fraction of samples poisoned on selected clients | `1.0` |
| Noise type | `--noise_type` / `-nt` | Noise-injection mode | `label_flipping` |

For 20 clients, `rc = 0.3` corresponds to 6 poisoned clients, and `rc = 0.5` corresponds to 10 poisoned clients, assuming the implementation uses the ratio to select that number of clients.

### 4.2 Asymmetric and symmetric noise

The parser accepts these exact case-sensitive choices:

```text
Asymmetric
Symmetric
```

The requested configurations are:

In this experiment configuration, `epsilon = 0.4` is the value used for `rs` in the Asymmetric and Symmetric scenarios.

| Scenario | `--noise_type` | `rc` | `rs` (called `epsilon` in the experiment notes) |
|---|---|---:|---:|
| Asymmetric noise | `Asymmetric` | `0.3` | `0.4` |
| Symmetric noise | `Symmetric` | `0.3` | `0.4` |

**Implementation note:** The supplied `main.py` does not define a CLI argument named `epsilon` or a dedicated `rs` argument for these two modes. It defines `--attack_noise_std` / `-ans` for Gaussian-noise standard deviation, so do not use that option as a substitute. Check the noise-injection implementation and pass `rs = 0.4` through the parameter expected by that code (or add an explicit CLI argument if needed).

The parser also accepts `gaussian_noise` and `None` as noise types.

## 5. Environment setup

Use the Python/Conda environment that contains the required packages for this repository. If the checkout includes `env_cuda_latest.yaml`, the following is an example; use the actual environment file and environment name in your clone:

```bash
conda env create -f env_cuda_latest.yaml
conda activate fl
```

Check the environment before running:

```bash
python3 --version
python3 -c "import numpy, sklearn, matplotlib, torch; print('Required imports OK')"
```

For GPU execution, also check:

```bash
nvidia-smi
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is unavailable, either install a compatible PyTorch build or run with `-dev cpu`.

## 6. `main.py`: configuration and execution

The supplied `main.py` has two key functions:

- `set_random_seed(seed)`: sets Python, NumPy, and PyTorch random seeds; configures deterministic cuDNN behavior.
- `run(args)`: determines the model dimensions from the selected dataset, creates the model and server for the chosen algorithm, trains the server, saves results, and reports average runtime.
- The `if __name__ == "__main__":` block: defines CLI arguments, parses them, converts the FedCorr `correction` and `fine_tuning` integer flags to booleans, sets the random seed and CUDA device, and calls `run(args)`.

### Common CLI arguments

| Option | Default in supplied `main.py` | Purpose |
|---|---:|---|
| `-algo`, `--algorithm` | `fedus` | Algorithm identifier |
| `-data`, `--dataset` | `cic_ids` | Dataset identifier |
| `-dev`, `--device` | `cuda` | `cpu` or `cuda` |
| `-did`, `--device_id` | `0` | CUDA device ID |
| `-dt`, `--data_type` | `noniid` | Data type/partition selector passed to the data-loading pipeline |
| `-nc`, `--num_clients` | `20` | Number of clients |
| `-lbs`, `--batch_size` | `4` | Local batch size |
| `-lr`, `--local_learning_rate` | `0.01` | Local learning rate |
| `-gr`, `--global_rounds` | `50` | Global rounds for algorithms using this argument |
| `-ls`, `--local_steps` | `5` | Local training steps |
| `-jr`, `--join_ratio` | `1.0` | Fraction of clients participating per round |
| `--seed` | `0` | Random seed |
| `--exp_name` | empty string | Experiment name |
| `-nt`, `--noise_type` | `label_flipping` | Noise type |
| `-npw`, `--num_of_poisoned_workers` | `0.3` | Poisoned-client ratio |
| `-psr`, `--poisoned_sample_ratio` | `1.0` | Poisoned-sample ratio |
| `-ans`, `--attack_noise_std` | `0` | Gaussian-noise standard deviation |

View the complete argument list for your checkout with:

```bash
python3 main.py --help
```

### FedCorr-specific arguments

| Option | Default | Meaning |
|---|---:|---|
| `--iteration1` | `5` | Stage 1 noise-detection outer iterations |
| `--rounds1` | `10` | Stage 2 fine-tuning rounds on clean clients |
| `--rounds2` | `40` | Stage 3 final FedAvg rounds |
| `--frac1` | `1.0` | Client fraction per sub-round in Stage 1 |
| `--frac2` | `1.0` | Client fraction in Stage 2 / Stage 3 |
| `--beta_fedcorr` | `0.0` | Proximal-term coefficient |
| `--relabel_ratio` | `0.5` | Fraction of estimated noisy samples to relabel |
| `--confidence_thres` | `0.5` | Minimum confidence for accepting a relabel |
| `--clean_set_thres` | `0.1` | Maximum estimated noise level for a clean client |
| `--correction` | `0` | `1` enables label correction; `0` disables it |
| `--fine_tuning` | `0` | `1` enables Stage 2 fine-tuning; `0` skips it |

The supplied source converts `--correction` and `--fine_tuning` from integers to booleans after parsing.

## 7. Running experiments

Run commands from the `system` directory, where `main.py` is located:

```bash
cd system
python3 main.py --help
```

### 7.1 FedCorr: label flipping, `rc = 0.3`, `rs = 1.0`

This command uses the FedCorr settings supplied for the experiments:

```bash
python3 -u main.py \
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
    --poisoned_sample_ratio 1.0 \
    --seed 0 \
    --exp_name nslkdd_fedcorr_label_flipping_rc0.3_rs1.0_seed0
```

For `rc = 0.5`, change the client ratio and experiment name:

```bash
--num_of_poisoned_workers 0.5
--poisoned_sample_ratio 1.0
--exp_name nslkdd_fedcorr_label_flipping_rc0.5_rs1.0_seed0
```

Repeat with seeds `0`, `1`, and `2`.

### 7.2 Other algorithms

A basic example for FedAvg is:

```bash
python3 -u main.py \
    -algo fedavg \
    -data nslkdd \
    -dev cpu \
    -nc 20 \
    -lbs 10 \
    -lr 0.03 \
    -gr 50 \
    -ls 5 \
    -dt noniid \
    --noise_type label_flipping \
    --num_of_poisoned_workers 0.3 \
    --poisoned_sample_ratio 1.0 \
    --seed 0 \
    --exp_name nslkdd_fedavg_label_flipping_rc0.3_rs1.0_seed0
```

Replace `-algo fedavg` with the registered identifier for the desired algorithm. Some algorithms may need additional algorithm-specific arguments. Use the `main.py --help` output and server implementation as the source of truth.

### 7.3 Selecting datasets

Replace `-data nslkdd` with one of:

```text
nslkdd
unsw
cic_ids
toniot
```

### 7.4 Selecting the partition

The requested partition names are `noniid20` (PAT) and `noniid20exdir` (ExDir, `alpha = 0.1`). The supplied parser only exposes `-dt/--data_type` with default `noniid`; it does not declare an explicit `--alpha` option for the general data partition. Ensure the dataset files are generated and the loader selects the intended partition. If the current data loader supports the partition through `-dt`, use the exact accepted value and document it in the experiment command. Do not assume `-dt noniid20exdir` works until it has been checked against the loader.

## 8. Experiment matrix

The planned experiment matrix is:

- **Datasets (4):** `nslkdd`, `unsw`, `cic_ids`, `toniot`
- **Partitions (2):** `noniid20` (PAT), `noniid20exdir` (ExDir, `alpha = 0.1`)
- **Algorithms (8):** `fedavg`, `fedgh`, `fedhome`, `fedrep`, `lg_fedavg`, `fedus` (FedPURE), `fedcorr`, `fednoro`
- **Attack settings (4):**
  1. Label flipping: `rc = 0.3`, `rs = 1.0`
  2. Label flipping: `rc = 0.5`, `rs = 1.0`
  3. Asymmetric noise: `rc = 0.3`, `rs = 0.4` (also denoted `epsilon = 0.4` in the experiment notes)
  4. Symmetric noise: `rc = 0.3`, `rs = 0.4` (also denoted `epsilon = 0.4` in the experiment notes)
- **Seeds (3):** `0`, `1`, `2`

If every combination is supported, the matrix contains `4 × 2 × 8 × 4 × 3 = 768` runs. The current `main.py` parser does not expose a dedicated `rs`/`epsilon` argument for the asymmetric and symmetric scenarios; check the noise-injection code and ensure `rs = 0.4` reaches the implementation before running them. Record any unsupported combinations rather than counting failed runs as completed.

## 9. Naming and reporting results

Use a unique experiment name containing the dataset, partition, algorithm, attack, and seed. Example:

```text
cic_ids_noniid20_fedcorr_label_flipping_rc0.3_rs1.0_seed0
```

Run each configuration with seeds `0`, `1`, and `2`, then report mean ± standard deviation. Keep the partition, attack parameters, evaluation procedure, and model settings consistent across algorithm comparisons.

## 10. Code structure

- `system/main.py`: command-line parser, random-seed setup, model/server selection, training entry point.
- `system/flcore/clients/`: client-side training implementations.
- `system/flcore/servers/`: server-side coordination and algorithm implementations.
- `system/flcore/trainmodel/models.py`: model architectures and backbones.
- `system/utils/data_utils.py`: dataset loading utilities.

Paths may vary slightly between repository versions; use the current checkout as the source of truth.
## Downloading and Preparing Datasets

**Note:** Raw dataset files are not stored in this GitHub repository. Download them from the official sources below and place them in the corresponding `dataset/<dataset_id>/rawdata/` directory.

### 1. Dataset Download Links

| Dataset | Dataset ID | Official Download / Source |
|---|---|---|
| UNSW-NB15 | `unsw` | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CIC-IDS2017 | `cic_ids` | https://www.unb.ca/cic/datasets/ids-2017.html |
| TON_IoT | `toniot` | https://research.unsw.edu.au/projects/toniot-datasets |
| NSL-KDD | `nslkdd` | https://www.unb.ca/cic/datasets/nsl.html |

For TON_IoT, use the **network traffic dataset**, not the telemetry or operating-system datasets. The NSL-KDD publisher page may no longer provide direct downloads; if using a mirror, verify the source and cite the original dataset.

### 2. Directory Structure

After downloading and extracting the datasets, organize the files as follows:

```text
dataset/
├── unsw/
│   └── rawdata/
├── cic_ids/
│   └── rawdata/
├── toniot/
│   └── rawdata/
└── nslkdd/
    └── rawdata/
