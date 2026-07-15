from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


EXE = os.path.expanduser("~/DLH/build/gen_groundtruth")
ER_PROBABILITIES = (0.00025, 0.00027, 0.0003)
LFR_MUS = (0.3, 0.4, 0.5)


def make_jobs(
    dataset_name: str,
    fvecs: str,
    max_elements: int = 1_000_000,
    topk: int = 10,
    k_query: int = 10_000,
    seed: int = 47,
    k_hop: int = 4,
    map_group_size: int = 20,
    metric: str = "l2",
    lfr_avg_degree: int = 15,
    lfr_max_degree: int = 30,
    lfr_degree_tau: float = 2.5,
    lfr_community_tau: float = 1.5,
    lfr_min_community: int = 20,
    lfr_max_community: int = 1000,
) -> list[dict]:
    size_label = f"{max_elements // 1_000_000}M" if max_elements % 1_000_000 == 0 else str(max_elements)
    common = {
        "fvecs": os.path.expanduser(fvecs),
        "max_elements": max_elements,
        "topk": topk,
        "k_query": k_query,
        "seed": seed,
        "k_hop_min": k_hop,
        "k_hop_max": k_hop,
        "map_group_size": map_group_size,
        "metric": metric,
    }

    jobs = []
    for prob in ER_PROBABILITIES:
        suffix = f"{prob:g}-{size_label}-{k_hop}hop-{map_group_size}group"
        jobs.append({
            **common,
            "name": f"{dataset_name}-{suffix}",
            "output_dir": f"~/test_data/{dataset_name}-{suffix}",
            "graph_model": "er",
            "prob": prob,
        })

    for mu in LFR_MUS:
        suffix = (
            f"LFR-k{lfr_avg_degree}-kmax{lfr_max_degree}-mu{mu:g}-"
            f"{size_label}-{k_hop}hop-{map_group_size}group"
        )
        jobs.append({
            **common,
            "name": f"{dataset_name}-{suffix}",
            "output_dir": f"~/test_data/{dataset_name}-{suffix}",
            "graph_model": "lfr",
            "lfr_avg_degree": lfr_avg_degree,
            "lfr_max_degree": lfr_max_degree,
            "lfr_degree_tau": lfr_degree_tau,
            "lfr_community_tau": lfr_community_tau,
            "lfr_mu": mu,
            "lfr_min_community": lfr_min_community,
            "lfr_max_community": lfr_max_community,
        })
    return jobs


def build_args(job: dict) -> list[str]:
    graph_model = job["graph_model"]
    out_dir = os.path.expanduser(job["output_dir"])
    args = [
        EXE,
        "--fvecs", os.path.expanduser(job["fvecs"]),
        "--max-elements", str(job["max_elements"]),
        "--output-dir", out_dir,
        "--graph-model", graph_model,
        "--topk", str(job["topk"]),
        "--k-query", str(job["k_query"]),
        "--seed", str(job["seed"]),
        "--k-hop-min", str(job["k_hop_min"]),
        "--k-hop-max", str(job["k_hop_max"]),
        "--map-group-size", str(job["map_group_size"]),
        "--metric", job["metric"],
    ]
    if graph_model == "er":
        args.extend(["--prob", str(job["prob"])])
    else:
        args.extend([
            "--lfr-avg-degree", str(job["lfr_avg_degree"]),
            "--lfr-max-degree", str(job["lfr_max_degree"]),
            "--lfr-degree-tau", str(job["lfr_degree_tau"]),
            "--lfr-community-tau", str(job["lfr_community_tau"]),
            "--lfr-mu", str(job["lfr_mu"]),
            "--lfr-min-community", str(job["lfr_min_community"]),
            "--lfr-max-community", str(job["lfr_max_community"]),
        ])
    return args


def run_job(job: dict) -> tuple[str, int]:
    name = job["name"]
    out_dir = os.path.expanduser(job["output_dir"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stdout_path = os.path.join(out_dir, f"stdout-{timestamp}.log")
    stderr_path = os.path.join(out_dir, f"stderr-{timestamp}.log")
    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        try:
            proc = subprocess.run(build_args(job), stdout=stdout, stderr=stderr, check=False)
            return name, proc.returncode
        except Exception as exc:
            stderr.write(f"[launcher] Exception: {exc!r}\n".encode("utf-8"))
            return name, -1


def run_jobs(jobs: list[dict], max_workers: int = 1) -> None:
    if not os.path.isfile(EXE) or not os.access(EXE, os.X_OK):
        raise SystemExit(f"Executable not found or not executable: {EXE}")

    max_workers = max(1, min(max_workers, len(jobs)))
    print(f"Running {len(jobs)} generation jobs with max_workers={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            name, code = future.result()
            print(f"[{'OK' if code == 0 else f'FAIL({code})'}] {name}")
            results.append((name, code))

    failures = [name for name, code in results if code != 0]
    if failures:
        print("Failed jobs: " + ", ".join(failures), file=sys.stderr)
        raise SystemExit(2)
    print("All generation jobs completed.")
