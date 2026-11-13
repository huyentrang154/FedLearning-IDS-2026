"""Run main.py jobs in parallel, longest jobs first, with resume support.

Job file format (one job per line, '#' starts a comment):
    <job_name> | <priority> | <arguments passed to main.py>

Higher priority starts first (use the expected run time, e.g. hours), so the
long FedPURE runs do not end up alone at the tail of the schedule.

Each job writes its console output to <log_dir>/<job_name>.log. A job is
skipped when <log_dir>/<job_name>.done exists, so re-running the same command
after an interruption only runs the missing jobs. Progress is appended to
<log_dir>/status.csv.

Example:
    python run_parallel.py --jobs jobs_nslkdd.txt --max-jobs 8 --log-dir logs/nslkdd
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def parse_jobs(path):
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, prio, args = [p.strip() for p in line.split("|", 2)]
            jobs.append((name, float(prio), args.split()))
    names = [j[0] for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        sys.exit(f"Duplicate job names: {sorted(dup)}")
    return sorted(jobs, key=lambda j: -j[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--max-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-procs", type=int, default=0,
                    help="also wait until fewer than this many main.py processes run on the "
                         "machine (counts other schedulers too); 0 = off")
    ap.add_argument("--min-free-mem-pct", type=int, default=0,
                    help="only start a job while the system-wide free memory is at least this "
                         "percentage (macOS memory_pressure); 0 = off")
    ap.add_argument("--launch-gap", type=float, default=1,
                    help="seconds to wait after starting a job, so its memory shows up before the next check")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    jobs = parse_jobs(args.jobs)
    todo = [j for j in jobs if not os.path.exists(os.path.join(args.log_dir, j[0] + ".done"))]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)} "
          f"with {args.max_jobs} in parallel", flush=True)

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    status_path = os.path.join(args.log_dir, "status.csv")
    lock = Lock()
    launch_lock = Lock()

    def running_main_procs():
        out = subprocess.run(["pgrep", "-f", "[Pp]ython.* -u (main|budget_ft)[.]py "], capture_output=True, text=True).stdout
        return len(out.split())

    def free_mem_pct():
        # macOS: "System-wide memory free percentage: NN%"; 100 (no limit) elsewhere
        try:
            out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
            return int(out.strip().splitlines()[-1].split(":")[-1].strip().rstrip("%"))
        except Exception:
            return 100

    def wait_for_slot():
        while args.max_procs and running_main_procs() >= args.max_procs:
            time.sleep(30)
        while args.min_free_mem_pct and free_mem_pct() < args.min_free_mem_pct:
            time.sleep(30)

    def run(job):
        name, _, job_args = job
        log_path = os.path.join(args.log_dir, name + ".log")
        with launch_lock:
            wait_for_slot()
            start = time.time()
            with lock:
                print(f"[START] {time.strftime('%H:%M:%S')} {name}", flush=True)
            log = open(log_path, "w")
            # a job whose arguments start with a .py file runs that script instead of main.py
            script, rest = (job_args[0], job_args[1:]) if job_args and job_args[0].endswith('.py') else ("main.py", job_args)
            proc = subprocess.Popen([args.python, "-u", script] + rest,
                                    stdout=log, stderr=subprocess.STDOUT, env=env)
            time.sleep(args.launch_gap)  # let pgrep / memory_pressure see the new process before the next launch
        rc = proc.wait()
        log.close()
        minutes = (time.time() - start) / 60
        if rc == 0:
            open(os.path.join(args.log_dir, name + ".done"), "w").close()
        with lock:
            new = not os.path.exists(status_path)
            with open(status_path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["finished", "job", "exit_code", "minutes"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, rc, f"{minutes:.1f}"])
            print(f"[{'DONE' if rc == 0 else 'FAIL'}] {time.strftime('%H:%M:%S')} {name} "
                  f"exit={rc} {minutes:.1f} min", flush=True)
        return rc

    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_jobs) as pool:
        for fut in as_completed([pool.submit(run, j) for j in todo]):
            failed += fut.result() != 0
    print(f"Finished. {len(todo) - failed} ok, {failed} failed. See {status_path}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
