from __future__ import annotations

import main_gen as yfcc
import main_gen_deep as deep
import main_gen_gist as gist
import main_gen_sift as sift
from main_gen_common import run_jobs


EXP_JOBS = [
    job
    for module in (sift, gist, deep, yfcc)
    for job in module.EXP_JOBS
    if job.get("graph_model") == "lfr"
]


if __name__ == "__main__":
    run_jobs(EXP_JOBS, max_workers=1)
