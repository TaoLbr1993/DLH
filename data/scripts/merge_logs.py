import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple


# Example line:
# [ACORN] ef=850 recall=99.424% time=32450 us
LINE_RE = re.compile(
    r"""^\[(?P<name>[^\]]+)\]\s+ef=(?P<ef>\d+)\s+recall=(?P<recall>[\d.]+)%\s+time=(?P<time>[\d.]+)\s*(?P<unit>ns|us|ms|s)\s*$"""
)
TS_IN_NAME_RE = re.compile(r"(\d{8})-(\d{6})")  # e.g. 20251221-160141


@dataclass(frozen=True)
class Record:
    baseline: str
    ef: int
    recall: float
    time_us: float
    qps: float
    src_file: str


def unit_to_us(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "ns":
        return value / 1000.0
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    if unit == "s":
        return value * 1_000_000.0
    raise ValueError(f"unsupported unit: {unit}")


def parse_timestamp_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = TS_IN_NAME_RE.search(base)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return int(dt.timestamp())


def file_recency_key(path: str) -> Tuple[int, float]:
    ts = parse_timestamp_from_filename(path)
    if ts is not None:
        return (ts, 0.0)
    try:
        return (0, os.path.getmtime(path))
    except OSError:
        return (0, 0.0)


def parse_file(path: str) -> List[Record]:
    out: List[Record] = []
    abspath = os.path.abspath(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            baseline = m.group("name").strip()
            ef = int(m.group("ef"))
            recall = float(m.group("recall"))
            t = float(m.group("time"))
            unit = m.group("unit")
            time_us = unit_to_us(t, unit)
            qps = (1_000_000.0 / time_us) if time_us > 0 else 0.0
            out.append(Record(baseline=baseline, ef=ef, recall=recall, time_us=time_us, qps=qps, src_file=abspath))
    return out


def choose_record(a: Record, b: Record, prefer: str) -> Record:
    """
    prefer:
      - latest: 选择更新的日志文件里的记录
      - best-recall: recall 高优先；recall 相同 qps 高优先
      - best-qps: qps 高优先；qps 相同 recall 高优先
    """
    if prefer == "latest":
        return b if file_recency_key(b.src_file) > file_recency_key(a.src_file) else a

    if prefer == "best-recall":
        if b.recall != a.recall:
            return b if b.recall > a.recall else a
        if b.qps != a.qps:
            return b if b.qps > a.qps else a
        return b

    if prefer == "best-qps":
        if b.qps != a.qps:
            return b if b.qps > a.qps else a
        if b.recall != a.recall:
            return b if b.recall > a.recall else a
        return b

    raise ValueError(f"unsupported prefer policy: {prefer}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge multiple stdout logs of a single baseline into *_stats.log (format identical to existing)."
    )
    ap.add_argument("--in-dir", type=str, required=True, help="Directory to scan recursively for stdout logs")
    ap.add_argument("--pattern", type=str, default="**/*-stdout-*.log", help="Glob pattern under --in-dir")
    ap.add_argument("--out", type=str, required=True, help="Output *_stats.log path")
    ap.add_argument("--prefer", type=str, default="latest", choices=["latest", "best-recall", "best-qps"],
                    help="How to resolve duplicated ef across multiple logs")
    args = ap.parse_args()

    paths = glob(os.path.join(args.in_dir, args.pattern), recursive=True)
    paths = [os.path.abspath(p) for p in paths if os.path.isfile(p)]
    if not paths:
        raise SystemExit("No input logs found. Check --in-dir/--pattern.")

    # Parse all records
    all_records: List[Record] = []
    for p in sorted(paths):
        all_records.extend(parse_file(p))

    if not all_records:
        raise SystemExit("No benchmark lines parsed. Expected lines like: [NAME] ef=.. recall=..% time=.. us")

    # Ensure single baseline
    baselines = sorted({r.baseline for r in all_records})
    if len(baselines) != 1:
        raise SystemExit(f"Expected a single baseline across logs, but found: {baselines}")
    baseline = baselines[0]

    # Merge by ef (union)
    best_by_ef: Dict[int, Record] = {}
    for r in all_records:
        if r.ef not in best_by_ef:
            best_by_ef[r.ef] = r
        else:
            best_by_ef[r.ef] = choose_record(best_by_ef[r.ef], r, prefer=args.prefer)

    efs = sorted(best_by_ef.keys())

    # Write stats log with EXACT format
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Benchmark Report\n")
        f.write("Search Times (ns):\n")
        f.write("Index \\ ef | " + " ".join([f"ef={ef}" for ef in efs]) + "\n")
        parts = []
        for ef in efs:
            r = best_by_ef[ef]
            # existing stats uses 3 decimals for recall and integer us for time
            parts.append(f"({r.recall:.3f}, {int(round(r.time_us))} us)")
        f.write(f"{baseline}: " + " ".join(parts) + "\n")

    print(f"[OK] baseline={baseline}, efs={len(efs)}")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()