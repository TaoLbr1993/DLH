from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext


# ----------------------------
# 1) 硬编码：需要读取哪些 logs 文件夹
# ----------------------------
LOG_DIRS: List[str] = [
    "logs/Deep10M-0.00027-2M-4hop-40group",
    "logs/Deep10M-0.00027-4M-4hop-80group",
    "logs/Deep10M-0.00027-6M-4hop-120group",
    "logs/Deep10M-0.00027-8M-4hop-160group",
    "logs/Deep10M-0.00027-10M-4hop-200group",
]

# ----------------------------
# 2) 硬编码：绘制哪些方法/基线，以及对应日志文件名
# ----------------------------
METHODS = [
    {
        "key": "DLH",
        "label": "DLH",
        "filename": "DLH_stats.log",
        "color": "tab:orange",
        "marker": "o",
    },
    {
        "key": "DLH-M",
        "label": "DLH-M",
        "filename": "DLH-M_stats.log",
        "color": "tab:red",
        "marker": "^",
    },
]

EFS: List[int] = [100, 250]

OUT_DIR = Path("figures/scala")
OUT_LEGEND = OUT_DIR / "scalability_legend.svg"
OUT_QPS = OUT_DIR / "scalability_qps.svg"
OUT_RECALL = OUT_DIR / "scalability_recall.svg"

# 视觉参数（按要求：线更细、点更小）
LINE_WIDTH = 1.8
MARKER_SIZE = 4.8
GRID_ALPHA = 0.35
GG_BG = "#E5E5E5"

# 让虚线更容易在 legend/图中辨认
DASH_PATTERN = (0, (4, 3))  # on, off


def _parse_dataset_size_m(folder_name: str) -> float:
    m = re.search(r"-(\d+(?:\.\d+)?)M(?:-|$)", folder_name)
    if not m:
        raise ValueError(f"无法从目录名解析数据集规模(M)：{folder_name}")
    return float(m.group(1))


def _to_seconds(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "s":
        return value
    if unit == "ms":
        return value * 1e-3
    if unit == "us":
        return value * 1e-6
    if unit == "ns":
        return value * 1e-9
    raise ValueError(f"未知时间单位：{unit}")


def _parse_stats_log(path: Path) -> Dict[int, Tuple[float, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    header = re.search(r"Index\s*\\\s*ef\s*\|\s*(.+)", text)
    if not header:
        raise ValueError(f"未找到 ef 表头：{path}")
    ef_list = [int(x) for x in re.findall(r"ef\s*=\s*(\d+)", header.group(1))]
    if not ef_list:
        raise ValueError(f"表头未解析到 ef=...：{path}")

    data_line = None
    for line in text.splitlines():
        s = line.strip()
        if ":" in s and "(" in s and ")" in s:
            low = s.lower()
            if low.startswith("benchmark"):
                continue
            if low.startswith("search times"):
                continue
            if low.startswith("index"):
                continue
            data_line = s
            break
    if data_line is None:
        raise ValueError(f"未找到数据行：{path}")

    pairs = re.findall(
        r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*(ns|us|ms|s)\s*\)",
        data_line,
        flags=re.I,
    )
    if not pairs:
        raise ValueError(f"未解析到 (recall, time) 对：{path}")

    if len(pairs) < len(ef_list):
        ef_list = ef_list[: len(pairs)]
    else:
        pairs = pairs[: len(ef_list)]

    out: Dict[int, Tuple[float, float]] = {}
    for ef, (rec_str, t_str, unit) in zip(ef_list, pairs):
        rec = float(rec_str)
        if rec > 1.0:
            rec /= 100.0
        t_sec = _to_seconds(float(t_str), unit)
        out[int(ef)] = (rec, t_sec)
    return out


def _collect_series(repo_root: Path) -> Dict[str, Dict[int, List[Tuple[float, float, float]]]]:
    data: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = {}
    for m in METHODS:
        data[m["key"]] = {ef: [] for ef in EFS}

    for d in LOG_DIRS:
        dpath = repo_root / d
        size_m = _parse_dataset_size_m(dpath.name)

        for m in METHODS:
            fpath = dpath / m["filename"]
            if not fpath.exists():
                print(f"[WARN] missing log: {fpath}")
                continue

            parsed = _parse_stats_log(fpath)
            for ef in EFS:
                if ef not in parsed:
                    print(f"[WARN] {fpath} has no ef={ef}")
                    continue
                recall, t_sec = parsed[ef]
                qps = (1.0 / t_sec) if t_sec > 0 else 0.0
                data[m["key"]][ef].append((size_m, recall, qps))

    for mk in data:
        for ef in data[mk]:
            data[mk][ef].sort(key=lambda x: x[0])
    return data


def _plot_on_axis(ax, data, *, value_idx: int) -> None:
    for m in METHODS:
        color = m["color"]
        marker = m["marker"]
        for ef in EFS:
            pts = data[m["key"]][ef]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[value_idx] for p in pts]
            ls = "-" if ef == EFS[0] else "--"
            ax.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                zorder=3,
            )


def _save_legend_svg() -> None:
    """
    仿照 bar_plot_gen.py 的 legend 生成方式：
    - 用 fig.legend(...)（不依赖 axes）
    - 用 legend 的 window_extent 做 bbox 裁剪并轻微 expanded
    - legend 框灰底 + 圆角 + 无边框描边
    - 画布保留少量白边，便于看出圆角
    """
    plt.style.use("ggplot")

    # 关键：画布用白底，legend 框用灰底，这样圆角轮廓才能“看得见”
    fig = plt.figure(figsize=(10.5, 0.75), facecolor="white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    handles: List[Line2D] = []
    labels: List[str] = []

    for m in METHODS:
        for ef in EFS:
            is_first = (ef == EFS[0])
            ls = "-" if is_first else DASH_PATTERN
            handles.append(
                Line2D(
                    [0], [0],
                    color=m["color"],
                    marker=m["marker"],
                    linestyle=ls,
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    dash_capstyle="butt",
                    solid_capstyle="butt",
                )
            )
            labels.append(f'{m["label"]} (b = {ef})')

    leg = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        fontsize=11.0,
        handlelength=3.2,
        numpoints=1,
        columnspacing=1.1,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    # legend 框：灰底 + 圆角 + 无描边
    frame = leg.get_frame()
    frame.set_facecolor(GG_BG)
    frame.set_edgecolor("none")
    frame.set_linewidth(0.0)
    frame.set_boxstyle("round,pad=0.22,rounding_size=0.2")  # rounding_size 调大更明显

    # 按 legend bbox 裁剪 + 稍微 expanded
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = leg.get_window_extent(renderer=renderer).transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1.02, 1.20)

    # 关键：保留一点白色留边（更容易看出圆角）
    fig.savefig(
        OUT_LEGEND,
        format="svg",
        bbox_inches=bbox,
        pad_inches=0.01,          # 适当白边
        facecolor=fig.get_facecolor(),  # 白底
    )
    plt.close(fig)
    print(f"[OK] saved: {OUT_LEGEND}")


def _save_qps_svg(data) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    _plot_on_axis(ax, data, value_idx=2)

    ax.set_xlabel("Dataset Size (×10$^6$)")
    ax.set_ylabel("QPS (/s)")

    # QPS：对数轴 + 主刻度固定为 10^1, 10^2, 10^3
    ax.set_yscale("log")
    ax.set_yticks([1e1, 1e2, 1e3])
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.minorticks_off()

    # 合理的 y 范围：至少覆盖 10^1..10^3，必要时向上扩展
    all_qps = []
    for m in METHODS:
        for ef in EFS:
            all_qps.extend([p[2] for p in data[m["key"]][ef]])
    max_qps = max(all_qps) if all_qps else 1e3
    upper_pow = max(3, int(math.ceil(math.log10(max_qps))) if max_qps > 0 else 3)
    ax.set_ylim(1e1, 10 ** upper_pow)

    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)

    xticks = [2, 4, 6, 8, 10]
    ax.set_xticks(xticks)

    fig.savefig(OUT_QPS, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] saved: {OUT_QPS}")


def _save_recall_svg(data) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    _plot_on_axis(ax, data, value_idx=1)

    ax.set_xlabel("Dataset Size (×10$^6$)")
    ax.set_ylabel("Recall Rate")
    ax.set_ylim(0.90, 1.00)
    ax.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)

    xticks = [2, 4, 6, 8, 10]
    ax.set_xticks(xticks)

    fig.savefig(OUT_RECALL, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] saved: {OUT_RECALL}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"输出目录不存在（按要求不创建）：{OUT_DIR}")

    data = _collect_series(repo_root)

    _save_legend_svg()
    _save_qps_svg(data)
    _save_recall_svg(data)


if __name__ == "__main__":
    main()