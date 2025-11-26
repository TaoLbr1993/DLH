import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# 硬编码：可执行文件路径
EXE = "/home/jiangyuntao/hnswlib-flex/build/gen_groundtruth"
# 硬编码：公共输入（可按需修改）
FVECs = "/home/jiangyuntao/Sift1M/sift_base.fvecs"

# 实验列表（每项一个独立 groundtruth 任务，注意保证 output_dir 唯一）
EXP_JOBS = [
    # {
    #     "name": "Sift1M-0.0002-20w-4hop",
    #     "fvecs": FVECs,
    #     "max_elements": 200000,
    #     "output_dir": "/home/jiangyuntao/test_data/Sift1M-0.0002-20w-4hop",
    #     "prob": 0.0002,
    #     "topk": 10,
    #     "k_query": 1000,
    #     "seed": 47,
    #     "k_hop_min": 4,
    #     "k_hop_max": 4,
    #     "map_group_size": 2,
    # },
    {
        "name": "Sift1M-0.00012-20w-4hop-2group",
        "fvecs": FVECs,
        "max_elements": 200000,
        "output_dir": "/home/jiangyuntao/test_data/Sift1M-0.00012-20w-4hop-2group",
        "prob": 0.00012,
        "topk": 10,
        "k_query": 1000,
        "seed": 47,
        "k_hop_min": 4,
        "k_hop_max": 4,
        "map_group_size": 2,
    },
    {
        "name": "Sift1M-0.00015-20w-4hop-2group",
        "fvecs": FVECs,
        "max_elements": 200000,
        "output_dir": "/home/jiangyuntao/test_data/Sift1M-0.00015-20w-4hop-2group",
        "prob": 0.00015,
        "topk": 10,
        "k_query": 1000,
        "seed": 47,
        "k_hop_min": 4,
        "k_hop_max": 4,
        "map_group_size": 2,
    },
    {
        "name": "Sift1M-0.00018-20w-4hop-2group",
        "fvecs": FVECs,
        "max_elements": 200000,
        "output_dir": "/home/jiangyuntao/test_data/Sift1M-0.00018-20w-4hop-2group",
        "prob": 0.00018,
        "topk": 10,
        "k_query": 1000,
        "seed": 47,
        "k_hop_min": 4,
        "k_hop_max": 4,
        "map_group_size": 2,
    },
    # {
    #     "name": "Sift1M-0.0001-20w-4hop-1group",
    #     "fvecs": FVECs,
    #     "max_elements": 200000,
    #     "output_dir": "/home/jiangyuntao/test_data/Sift1M-0.0001-20w-4hop-1group",
    #     "prob": 0.0001,
    #     "topk": 10,
    #     "k_query": 1000,
    #     "seed": 47,
    #     "k_hop_min": 4,
    #     "k_hop_max": 4,
    #     "map_group_size": 1,
    # },
    # {
    #     "name": "Sift1M-0.0001-10w-4hop-1group",
    #     "fvecs": FVECs,
    #     "max_elements": 100000,
    #     "output_dir": "/home/jiangyuntao/test_data/Sift1M-0.0001-10w-4hop-1group",
    #     "prob": 0.0001,
    #     "topk": 10,
    #     "k_query": 1000,
    #     "seed": 47,
    #     "k_hop_min": 4,
    #     "k_hop_max": 4,
    #     "map_group_size": 1,
    # },
]

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def run_job(job: dict) -> tuple[str, int]:
    name = job["name"]
    out_dir = job["output_dir"]

    ensure_dir(out_dir)

    # 独立日志
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stdout_path = os.path.join(out_dir, f"stdout-{ts}.log")
    stderr_path = os.path.join(out_dir, f"stderr-{ts}.log")

    args = [
        EXE,
        "--fvecs", job["fvecs"],
        "--max-elements", str(job["max_elements"]),
        "--output-dir", out_dir,
        "--prob", str(job["prob"]),
        "--topk", str(job["topk"]),
        "--k-query", str(job["k_query"]),
        "--seed", str(job["seed"]),
        "--k-hop-min", str(job["k_hop_min"]),
        "--k-hop-max", str(job["k_hop_max"]),
        "--map-group-size", str(job["map_group_size"]),
    ]

    env = os.environ.copy()
    # 不限制内部库线程（按库默认）
    # env["OMP_NUM_THREADS"] = str(job.get("omp_threads", 1))
    # env["MKL_NUM_THREADS"] = str(job.get("omp_threads", 1))
    # env["OPENBLAS_NUM_THREADS"] = str(job.get("omp_threads", 1))

    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        try:
            proc = subprocess.run(args, stdout=out, stderr=err, env=env, check=False)
            return name, proc.returncode
        except Exception as e:
            err.write(f"[launcher] Exception: {repr(e)}\n".encode("utf-8"))
            return name, -1

def main():
    if not os.path.isfile(EXE) or not os.access(EXE, os.X_OK):
        print(f"错误：找不到可执行文件或无执行权限: {EXE}", file=sys.stderr)
        sys.exit(1)

    # 并发度：资源充足 -> 全部并行
    max_workers = len(EXP_JOBS)
    print(f"并发执行 {len(EXP_JOBS)} 个任务，max_workers={max_workers}")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_job, job) for job in EXP_JOBS]
        for fut in as_completed(futs):
            name, code = fut.result()
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"[{status}] {name}")
            results.append((name, code))

    # 汇总
    fails = [n for n, c in results if c != 0]
    if fails:
        print("失败任务：", ", ".join(fails))
        sys.exit(2)
    print("全部任务完成。")

if __name__ == "__main__":
    main()