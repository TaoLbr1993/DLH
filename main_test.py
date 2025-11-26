import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

ef_list = [
    "1,2,4,8,10,15,20,25,30,35,40,45,50,60,80,100,120,140,160,180,200,220,240,260,280,300",
    "1,2,4,8,10,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300"
]

# 1. 数据集配置
datasets = [
    # {
    #     "dataset_name": "Sift1M-0.0001-20w-4hop-2group",
    #     "data_dir": "/home/jiangyuntao/test_data/Sift1M-0.0001-20w-4hop-2group",
    #     "log_dir": "/home/jiangyuntao/hnswlib-flex/logs/Sift1M-0.0001-20w-4hop-2group",
    #     "acorn_gamma": "10",
    # },
    # {
    #     "dataset_name": "Sift1M-0.00012-20w-4hop-2group",
    #     "data_dir": "/home/jiangyuntao/test_data/Sift1M-0.00012-20w-4hop-2group",
    #     "log_dir": "/home/jiangyuntao/hnswlib-flex/logs/Sift1M-0.00012-20w-4hop-2group",
    #     "acorn_gamma": "5",
    # },
    # {
    #     "dataset_name": "Sift1M-0.00015-20w-4hop-2group",
    #     "data_dir": "/home/jiangyuntao/test_data/Sift1M-0.00015-20w-4hop-2group",
    #     "log_dir": "/home/jiangyuntao/hnswlib-flex/logs/Sift1M-0.00015-20w-4hop-2group",
    #     "acorn_gamma": "3",
    # },
    {
        "dataset_name": "Sift1M-0.00018-20w-4hop-2group",
        "data_dir": "/home/jiangyuntao/test_data/Sift1M-0.00018-20w-4hop-2group",
        "log_dir": "/home/jiangyuntao/hnswlib-flex/logs/Sift1M-0.00018-20w-4hop-2group",
        "acorn_gamma": "2",
    },
    {
        "dataset_name": "Sift1M-0.0002-20w-4hop-2group",
        "data_dir": "/home/jiangyuntao/test_data/Sift1M-0.0002-20w-4hop-2group",
        "log_dir": "/home/jiangyuntao/hnswlib-flex/logs/Sift1M-0.0002-20w-4hop-2group",
        "acorn_gamma": "1",
    },
]

# 2. 基线算法配置
#    - name: 算法名称
#    - exe_name: 对应的可执行文件名
#    - ef_list_idx: 使用的 ef_list 索引
#    - extra_argv: (可选) 额外的命令行参数
baselines_config = [
    {"name": "GHNSW-V6", "exe_name": "baseline_ghnsw_v6", "ef_list_idx": 0},
    {"name": "GHNSW-V5", "exe_name": "baseline_ghnsw_v5", "ef_list_idx": 0},
    {"name": "HNSW", "exe_name": "baseline_hnsw", "ef_list_idx": 0},
    {"name": "ACORN", "exe_name": "baseline_acorn", "ef_list_idx": 1},
    {"name": "NAVIX", "exe_name": "baseline_navix", "ef_list_idx": 1},
]

# 3. 动态生成 EXP_JOBS
def generate_exp_jobs(datasets_to_run, baselines_to_run):
    jobs = []
    build_dir = "/home/jiangyuntao/hnswlib-flex/build"
    for dataset_cfg in datasets_to_run:
        for baseline_cfg in baselines_to_run:
            job_name = f"{baseline_cfg['name']}_{dataset_cfg['dataset_name']}"
            
            argv = [
                "--data-dir", dataset_cfg["data_dir"],
                "--out", dataset_cfg["log_dir"],
                "--ef-list", ef_list[baseline_cfg["ef_list_idx"]],
            ]
            
            # 为 ACORN 添加特定的、与数据集相关的参数
            if baseline_cfg["name"] == "ACORN" and "acorn_gamma" in dataset_cfg:
                argv.extend(["--acorn-gamma", dataset_cfg["acorn_gamma"]])

            if "extra_argv" in baseline_cfg:
                argv.extend(baseline_cfg["extra_argv"])

            jobs.append({
                "name": job_name,
                "exe": os.path.join(build_dir, baseline_cfg["exe_name"]),
                "out_dir": dataset_cfg["log_dir"],
                "argv": argv,
            })
    return jobs

# 从配置中生成所有任务
EXP_JOBS = generate_exp_jobs(datasets, baselines_config)


def run_job(job: dict) -> tuple[str, int]:
    name = job["name"]
    exe = job["exe"]
    out_dir = job["out_dir"]
    argv = job["argv"]

    ensure_dir(out_dir)

    baseline_name = name.split("_")[0]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stdout_path = os.path.join(out_dir, f"{baseline_name}-stdout-{ts}.log")
    stderr_path = os.path.join(out_dir, f"{baseline_name}-stderr-{ts}.log")

    args = [exe] + list(argv)
    env = os.environ.copy()

    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        try:
            proc = subprocess.run(args, stdout=out, stderr=err, env=env, check=False)
            return name, proc.returncode
        except Exception as e:
            err.write(f"[launcher] Exception: {repr(e)}\n".encode("utf-8"))
            return name, -1

def main():
    # 校验每个任务的可执行文件
    bad = []
    for job in EXP_JOBS:
        exe = job.get("exe")
        if not exe or not os.path.isfile(exe) or not os.access(exe, os.X_OK):
            bad.append((job["name"], exe))
    if bad:
        for n, e in bad:
            print(f"错误：[{n}] 找不到可执行文件或无执行权限: {e}", file=sys.stderr)
        sys.exit(1)

    max_workers = len(EXP_JOBS)  # 资源足够 -> 全部并行
    print(f"并发执行 {len(EXP_JOBS)} 个 baseline 任务，max_workers={max_workers}")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_job, job) for job in EXP_JOBS]
        for fut in as_completed(futs):
            name, code = fut.result()
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"[{status}] {name}")
            results.append((name, code))

    fails = [n for n, c in results if c != 0]
    if fails:
        print("失败任务：", ", ".join(fails))
        sys.exit(2)
    print("全部 baseline 任务完成。")

if __name__ == "__main__":
    main()