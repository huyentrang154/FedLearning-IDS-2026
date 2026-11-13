# Đo thời gian huấn luyện (cột "Training time" trong bảng chi phí)

Tài liệu này hướng dẫn chạy thí nghiệm đo thời gian huấn luyện của 8 phương pháp. Kết quả dùng để điền cột **Training time (s)** trong bảng chi phí gộp của bài báo (`\label{tab:comm-cost}\label{tab:time-cost}`). Thí nghiệm này trả lời ý của Reviewer 2 về chi phí tính toán.

---

## 1. Đại lượng cần đo

**Training time** là thời gian thực (wall-clock), tính bằng giây, để hoàn tất **toàn bộ** quá trình huấn luyện của một phương pháp. Số này phải đo đúng như trong một hệ FL thật:

- **20 client huấn luyện song song**: trong mỗi vòng, server chờ client chậm nhất xong rồi mới tổng hợp.
- **Các phương pháp chạy lần lượt**, mỗi lần chỉ một phương pháp, để chúng không tranh CPU với nhau.

Số này **không phải** thời gian huấn luyện cục bộ của một client. Nó cũng **không phải** tổng thời gian của 20 client cộng lại.

## 2. Vì sao không chạy 20 tiến trình client thật

Code mô phỏng FL chạy các client **lần lượt trong một tiến trình**. Nếu chạy 20 tiến trình thật trên một máy 10 nhân, các client sẽ tranh CPU và bộ nhớ, nên số đo bị phồng lên và phụ thuộc vào máy.

Vì vậy ta **đo thời gian của từng client** rồi **tính ra thời gian song song**:

```
thời gian song song = thời gian server + Σ_(mỗi nhóm) max(thời gian các client trong nhóm)
```

- Một **nhóm** là phần việc của các client nằm giữa hai lần server phải chờ (nhận mô hình hoặc tổng hợp). Các client trong cùng một nhóm có thể chạy song song trong hệ thật. Với đa số phương pháp, mỗi vòng là một nhóm.
- Nếu một client làm nhiều việc trong cùng nhóm (ví dụ `train` rồi `collect_protos`), thời gian các việc đó được cộng lại.
- **Thời gian server** = tổng thời gian − thời gian đánh giá − thời gian ghi chép − tổng thời gian của các client. Phần này gồm gửi/nhận mô hình, tổng hợp, huấn luyện ở server (ví dụ `train_predictor`), phát hiện client nhiễu (GMM của FedCorr, FedNoRo), v.v.
- **Không tính phần đánh giá** (`evaluate`, `_evaluate_and_log`): đây là phần đo đạc, không thuộc quá trình huấn luyện. Việc client làm trong lúc đánh giá cũng không được tính.
- **Không tính phần ghi chép của code thí nghiệm** (`track_best`, `round_models`, `write_final_summary`, `save_model_artifacts`, `save_results`). Các hàm này sao chép/lưu mô hình mỗi vòng và ghi log; một hệ FL thật không làm việc này.
- **Không tính** phần nạp dữ liệu và khởi tạo mô hình trước `server.train()`.

Các trường hợp đặc biệt được tính đúng theo cách thuật toán chạy:

- **FedCorr Stage 1** (iteration1 = 5, `frac1 = 1/N`): server tổng hợp sau **từng client**, nên mỗi nhóm chỉ có 1 client và giai đoạn này vẫn tuần tự, đúng như thuật toán gốc. Các giai đoạn sau có nhóm nhiều client.
- **LID của FedCorr** (`_lid_term`): trong code mô phỏng hàm này nằm ở server, nhưng mỗi client tính LID trên đầu ra của chính nó. Vì vậy thời gian của hàm được tính cho client vừa huấn luyện ngay trước đó.
- **FedPURE**: Phase 1 có 49 vòng, mỗi vòng 5 epoch AE. Vòng cuối là Phase 2 với 2500 epoch CE. Vòng cuối là một nhóm riêng và chiếm phần lớn thời gian.

## 3. Các file code liên quan

Thư mục làm việc: `FedLearning-IDS-2026/system/`.

| File | Vai trò |
|---|---|
| `utils/timing.py` | `FLTimer` và `install(server)`. Bọc (wrap) các hàm của client, server và hàm đánh giá để đo thời gian. **Chỉ đo, không thay đổi thuật toán.** |
| `main.py` | Tham số `--timing_out <file.json>`. Khi có tham số này, gọi `install(server)` trước `server.train()` và ghi JSON sau khi train xong. Khi không có, code chạy như cũ. |
| `jobs_timing_nsl.txt` | 8 job, mỗi phương pháp một job. |
| `run_parallel.py` | Bộ lập lịch. Đặt `--max-jobs 1 --max-procs 1` để chạy từng job một. |
| `summarize_timing.py` | Đọc các file JSON và in bảng kèm các dòng LaTeX. |

Các hàm được bọc:

- Việc của client: `train`, `collect_protos`, `get_outputs_and_loss`, `compute_per_class_losses`, `relabel`.
- Điểm chờ của server (kết thúc một nhóm): `receive_models`, `receive_protos`, `aggregate_parameters`, `train_predictor`, và hàm `_fedavg` ở cấp module (FedCorr dùng hàm này).
- Đánh giá (bị loại khỏi phép đo): `evaluate`, `_evaluate_and_log`.
- Ghi chép (bị loại khỏi phép đo, không kết thúc nhóm): `track_best`, `round_models`, `write_final_summary`, `save_model_artifacts`, `save_results`.
- Phần việc của client nằm ở hàm cấp module (tính cho client vừa chạy trước đó): `_lid_term` của FedCorr.

## 4. Thiết lập thí nghiệm

| Mục | Giá trị |
|---|---|
| Dữ liệu | NSL-KDD, chia `pat` (`-dt noniid`), không nhiễu (`-nt none -npw 0 -psr 0`) |
| Seed | 0 |
| Thiết bị | CPU (`-dev cpu`) |
| Phương pháp | FedAvg, FedGH, FedRep, FedHome, LG-FedAvg, FedNoRo, FedCorr, FedPURE (`fedus`) |
| Siêu tham số | Mặc định của repo, giống các bảng kết quả chính. FedCorr dùng bản đã sửa (`--iteration1 5 --rounds1 10 --rounds2 40`, là mặc định). FedPURE là bản gốc. |
| Kết quả | `../results/timing/<algo>.json` |

Mỗi dòng trong `jobs_timing_nsl.txt` có dạng `tên | ưu tiên | tham số main.py`, ví dụ:

```
timing_nsl_pat_fedavg | 9 | -algo fedavg -data nslkdd -dt noniid -dev cpu -nt none -npw 0 -psr 0 --seed 0 --exp_name timing --timing_out ../results/timing/fedavg.json
```

### Yêu cầu về máy (quan trọng)

1. **Chỉ chạy một job tại một thời điểm**, và không có job huấn luyện nào khác trên máy. Nếu có tải khác, số đo sẽ cao hơn thực tế và các phương pháp không so sánh được với nhau.
2. **Chạy cả 8 phương pháp trên cùng một máy.** Nếu phải chạy lại một phương pháp trên máy khác, hãy chạy lại cả 8.
3. Cắm điện, tắt chế độ ngủ (trên macOS dùng `caffeinate -i`), đóng các ứng dụng nặng.
4. Ghi lại cấu hình máy (CPU, số nhân, RAM, hệ điều hành, phiên bản PyTorch) để đưa vào chú thích bảng.

## 5. Cách chạy

```bash
cd FedLearning-IDS-2026/system
source ~/.venvs/fedlearning-ids/bin/activate   # hoặc môi trường Python của bạn
```

Chạy cả 8 job lần lượt. Job đã xong có file `.done` trong `logs/timing` và sẽ được bỏ qua, nên nếu bị ngắt giữa chừng thì chạy lại đúng lệnh này để tiếp tục:

```bash
caffeinate -i python run_parallel.py --jobs jobs_timing_nsl.txt --max-jobs 1 --max-procs 1 --launch-gap 5 --log-dir logs/timing
```

Để chạy riêng một phương pháp (trên Linux, bỏ `caffeinate -i`):

```bash
python -u main.py -algo fedavg -data nslkdd -dt noniid -dev cpu -nt none -npw 0 -psr 0 --seed 0 --exp_name timing --timing_out ../results/timing/fedavg.json
```

Thư mục `../results/timing/` được tạo tự động.

**Thời gian dự kiến**: khoảng 16–18 giờ cho cả 8 job trên máy M1 Pro 10 nhân. FedPURE chiếm phần lớn thời gian vì có 2500 epoch ở Phase 2. Các phương pháp khác mất khoảng vài chục phút đến một giờ mỗi phương pháp.

### Chạy thử nhanh (khuyến nghị trước khi chạy thật)

Chạy thử vài vòng để kiểm tra môi trường. Dùng `--exp_name` và file JSON riêng, rồi xoá kết quả chạy thử sau khi kiểm tra:

```bash
python -u main.py -algo fedus -data nslkdd -dt noniid -dev cpu -nt none -npw 0 -psr 0 --seed 0 -gr 3 --exp_name smoke_timing --timing_out /tmp/smoke_fedus.json
```

`-gr 3` là số vòng toàn cục (mặc định 50). Cuối log phải có dòng `[timing] sequential ...s, parallel ...s, eval ...s, bookkeeping ...s, groups ... -> ...`.

## 6. Kết quả và cách kiểm tra

Mỗi file JSON có các trường sau:

| Trường | Ý nghĩa |
|---|---|
| `parallel_s` | **Số cần điền vào bảng**: thời gian khi 20 client chạy song song |
| `sequential_s` | Thời gian nếu các client chạy lần lượt (= tổng − đánh giá − ghi chép) |
| `server_s` | Phần thời gian ở server |
| `client_work_s` | Tổng thời gian của tất cả client |
| `eval_s` | Thời gian đánh giá (đã loại khỏi các số trên) |
| `bookkeeping_s` | Thời gian ghi chép: lưu mô hình, ghi log (đã loại khỏi các số trên) |
| `total_wall_s` | Tổng thời gian thực của `server.train()` |
| `n_groups`, `clients_per_group_min/max` | Số nhóm và số client mỗi nhóm |
| `groups` | Thời gian từng client trong từng nhóm (để kiểm tra chi tiết) |

Các điểm cần kiểm tra:

- `parallel_s` ≤ `sequential_s`, và `sequential_s` + `eval_s` + `bookkeeping_s` = `total_wall_s`.
- `clients_per_group_max` bằng số client tham gia một vòng (20 với cấu hình mặc định).
- Với FedCorr, `clients_per_group_min` = 1 (Stage 1 tuần tự).
- Với FedPURE, nhóm cuối (Phase 2) lớn hơn hẳn các nhóm khác.
- Không có job nào khác chạy cùng lúc. Xem log trong `logs/timing/`.

## 7. Tổng hợp và điền vào bài báo

```bash
python summarize_timing.py            # mặc định đọc ../results/timing
```

Script in bảng `parallel / sequential / server / eval / bookkeeping / groups` cho từng phương pháp. Sau đó nó in các dòng LaTeX `Method & <parallel_s> \\`.

Điền giá trị `parallel_s` vào cột **Training time (s)** của bảng chi phí gộp. Các số đang có trong bảng (953, 1088, 816, 1347, 985, 1237, 975, 12643) chỉ là **ước tính tạm**, phải thay bằng số đo được.

Gợi ý câu mô tả cho chú thích bảng hoặc phần Experimental setup:

> Training time is the wall-clock time to complete the whole training run on NSL-KDD (pathological split, seed 0), measured on <CPU / RAM / OS>. Client updates are timed individually and combined as if the 20 clients trained in parallel: in every round, the server waits for the slowest client. Server-side computation is included; evaluation and experiment bookkeeping (checkpointing, logging) are excluded.

## 8. Lưu ý

- `--timing_out` chỉ thêm phần đo đạc. Độ chính xác và F1 của các run này giống run thường, nên kết quả F1 có thể đối chiếu với Bảng 3 (NSL pat, seed 0) như một bước kiểm tra phụ.
- Phần đo đạc có overhead không đáng kể: mỗi lần gọi hàm chỉ thêm một cặp `time.perf_counter()`.
- Không so sánh các số này với thời gian của các run cũ chạy song song nhiều job, vì các run đó bị tranh tài nguyên.
- Nếu có câu hỏi về định nghĩa hoặc cách tính, xem docstring đầu file `utils/timing.py`.
