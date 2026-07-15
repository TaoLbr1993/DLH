from __future__ import annotations

from main_gen_common import make_jobs, run_jobs


EXP_JOBS = make_jobs(
    dataset_name="Sift1M",
    fvecs="~/Sift1M/sift_base.fvecs",
)


if __name__ == "__main__":
    run_jobs(EXP_JOBS, max_workers=1)
