#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite benchmarking plots with a single top‑level CONFIG.

中英双语说明 / Bilingual notes
--------------------------------
- 你可以在 CONFIG 里一次性配置所有算法的颜色、marker、markersize、显示名称、
  日志文件名模式、是否启用、以及绘图的全局参数（风格、坐标范围、ticks 等）。
- 所有生成逻辑（单图、三合一、legend）都从 CONFIG 读取，便于统一改动与复用。
- 若某算法的日志不存在，则自动跳过（不会报错）。

Key ideas
---------
- Single source of truth: CONFIG.
- Algorithms are registered in CONFIG["ALGORITHMS"].
- Orders per mode (range/tag) live in CONFIG["ORDERS"].
- Dataset‑specific y‑ranges live in CONFIG["PLOTS"]["dataset_settings"].

依赖 / Dependencies
--------------------
- figure_tools_pro: create_panel_layout, draw_free_lines, save_fig, save_legend_only_from_styles
- matplotlib
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import os
import re
import matplotlib.pyplot as plt

# Make local package available (keep user intent)
os.chdir(os.path.dirname(__file__))
sys.path.append("..")
from figure_tools_pro import (
    create_panel_layout,
    draw_free_lines,
    save_fig,
)
# save_legend_only_from_styles is imported lazily in gen_legend()

# ================================================================
#                           CONFIG
# ================================================================
CONFIG: Dict = {
    # -------- Paths & style --------
    "PATHS": {
        "LOG_ROOT": "logs",          # root for logs/<group>/<file>
        "FIG_ROOT": "figures",       # output figures
    },
    "STYLE": {
        "matplotlib_style": "ggplot",    # e.g., 'ggplot', 'default'
    },

    # -------- Palette (centralized) --------
    # You can tweak these once and reuse in ALGORITHMS
    "PALETTE": {
        "GHNSW": "#E24A33",
        "HNSW": "#348ABD",
        "ACORN": "#988ED5",
    },

    # -------- Algorithm registry --------
    # key: short stable key used by ORDERS, legend, and generators
    # value: display & file options
    "ALGORITHMS": {
        "HNSW": {
            "label": "HNSW + Filter",
            "filenames": ["HNSW.log"],
            "color": "{PALETTE.HNSW}",
            "marker": "s",
            "markersize": 7,
            "line_width": 3,
            "markeredgewidth": 1.2,
            "visible": True,
        },
        "ACORN": {
            "label": "ACORN",
            "filenames": ["ACORN.log"],
            "color": "{PALETTE.ACORN}",
            "marker": "^",
            "markersize": 8,
            "line_width": 3,
            "markeredgewidth": 1.0,
            "visible": True,
        }
    },

    # Which algorithms to draw per mode (and their ordering/z‑order precedence)
    "ORDERS": {
        "range": [
            "HNSW", "ACORN",
        ],
        "tag": [
            "HNSW", "ACORN",
        ],
        # Legend order (if you want a global legend); defaults to the union of above
        "legend": [
            "HNSW", "ACORN",
        ],
    },

    # Global plotting controls
    "PLOTS": {
        # dataset -> mode -> (y_min, y_max)
        "dataset_settings": {
            "RANDOM": {"range": (2, 2000),    "tag": (2, 2000 )},
            "SIFT1M": {"range": (20, 9000),     "tag": (20, 9000)},
            "WIT":    {"range": (2, 2000 ),     "tag": (2, 2000 )},
            "YFCC":   {"range": (2, 4000 ),     "tag": (2, 4000 )},
        },
        "x_range": (0.79, 1.01),
        "x_ticks": [0.8, 0.85, 0.9, 0.95, 1.0],
        "use_log_y": True,
        "ytick_multipliers": (1,),   # or (1, 2, 5)
        "line_style": "-",
        "default_line_width": 3,
        "default_markeredgewidth": 1.0,
        "figsize_composite": (20, 4.5),
        "legend": {
            "ncol": 10,
            "linewidth": 2,
            "markersize": None,  # if None: from algorithms
            "markeredgewidth": 0.5,
            "columnspacing": 2.0,
            "outfile": "legend_ggplot.svg",
        },
        "use_pareto": True,  # True to filter dominated points per line
    },
}

# ---------- Utility to resolve palette references inside strings ----------

def _resolve_palette_placeholders():
    pal = CONFIG["PALETTE"]
    for key, spec in CONFIG["ALGORITHMS"].items():
        color = spec.get("color", "")
        if isinstance(color, str) and color.startswith("{") and color.endswith("}"):
            # format: {PALETTE.KEY}
            inner = color.strip("{}")
            parts = inner.split(".")
            if len(parts) == 2 and parts[0] == "PALETTE":
                spec["color"] = pal.get(parts[1], color)

_resolve_palette_placeholders()

# ================================================================
#                        Parsing helpers
# ================================================================

def parse_search_times(text: str) -> List[Tuple[float, float]]:
    """
    Parse the table block "Search Times (ns)" and return [(recall, time_us), ...].
    - Only the first method line in the table is used (per-file contains one method).
    - Units in pairs are assumed 'us' (microseconds), as in current logs.
    """
    m = re.search(r"Search Times\s*\(ns\)\s*:\s*\n(.*?)(?:\n-[-]+\n|\Z)", text, re.S)
    if not m:
        return []
    section = m.group(1)

    # optional: parse ef header if you want to align by length
    # ef_list = [int(x) for x in re.findall(r"ef=(\d+)", section)]

    pair_pattern = re.compile(r"\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]+)\s*us\s*\)")
    for line in section.splitlines():
        line_strip = line.strip()
        if not line_strip or line_strip.startswith("Index \\ ef"):
            continue
        if "(" not in line:
            continue
        pairs = pair_pattern.findall(line)
        if not pairs:
            continue
        pairs_f = [(float(a), float(b)) for a, b in pairs]
        return pairs_f
    return []


def parse_search_times_file(path: str) -> List[Tuple[float, float]]:
    p = Path(path)
    if p.exists():
        return parse_search_times(p.read_text(encoding="utf-8", errors="ignore"))
    # silent miss
    return []

# ================================================================
#                         Plot helpers
# ================================================================

def _pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Keep only non‑dominated points (x↑, y↑ better): remove points that are dominated.
    x = recall * 100, y = QPS.
    """
    if not points:
        return []
    pts = sorted(points, key=lambda p: (-p[0], -p[1]))
    out: List[Tuple[float, float]] = []
    max_y = -float("inf")
    for x, y in pts:
        if y > max_y:
            out.append((x, y))
            max_y = y
    return out


def _resolve_file_for_algo(group_name: str, filenames: Iterable[str]) -> Optional[str]:
    root = CONFIG["PATHS"]["LOG_ROOT"]
    for fname in filenames:
        p = Path(root) / group_name / fname
        if p.exists():
            return str(p)
    # fallback to first even if missing; caller will handle empty result
    return str(Path(root) / group_name / next(iter(filenames))) if filenames else None


def _load_lines_for_group(group_name: str, algo_keys: Iterable[str]):
    """
    Load curves for a group (e.g., 'WIT-range-4p') using the algorithms in algo_keys.
    - Convert (recall, time_us) -> (recall*100, QPS)
    - Optional Pareto filtering (CONFIG['PLOTS']['use_pareto'])
    Returns: (lines, labels, colors, markers, zorders, markersizes, line_widths, markeredgewidths)
    """
    lines: List[List[Tuple[float, float]]] = []
    labels: List[str] = []
    colors: List[str] = []
    markers: List[str] = []
    markersizes: List[float] = []
    line_widths: List[float] = []
    markeredgewidths: List[float] = []

    use_pareto = CONFIG["PLOTS"].get("use_pareto", False)

    for key in algo_keys:
        spec = CONFIG["ALGORITHMS"].get(key)
        if not spec or not spec.get("visible", True):
            continue
        fpath = _resolve_file_for_algo(group_name, spec.get("filenames", []))
        raw_pairs = parse_search_times_file(fpath) if fpath else []
        converted = [(r, 1_000_000.0 / t_us) for (r, t_us) in raw_pairs]
        if use_pareto:
            converted = _pareto_front(converted)
        if not converted:
            # skip algorithms without data silently
            continue

        lines.append(converted)
        labels.append(spec.get("label", key))
        colors.append(spec.get("color"))
        markers.append(spec.get("marker", "o"))
        markersizes.append(spec.get("markersize", CONFIG["PLOTS"].get("default_markersize", 7)))
        line_widths.append(spec.get("line_width", CONFIG["PLOTS"].get("default_line_width", 3)))
        markeredgewidths.append(spec.get("markeredgewidth", CONFIG["PLOTS"].get("default_markeredgewidth", 1.0)))

    # Higher priority (earlier in algo_keys) gets higher zorder
    zorders = list(range(len(lines), 0, -1))
    return lines, labels, colors, markers, zorders, markersizes, line_widths, markeredgewidths


def _nice_log_ticks(y_min: float, y_max: float, multipliers: Iterable[int] = (1,)) -> List[float]:
    """Generate nice log‑scale ticks within [y_min, y_max]. Choose multipliers like (1,2,5)."""
    if y_min <= 0:
        y_min = 1e-6
    import math
    k_min = int(math.floor(math.log10(y_min)))
    k_max = int(math.ceil(math.log10(y_max)))
    ticks: List[float] = []
    for k in range(k_min, k_max + 1):
        for m in multipliers:
            v = m * (10 ** k)
            if y_min <= v <= y_max:
                ticks.append(v)
    if ticks and ticks[0] > y_min:
        ticks.insert(0, y_min)
    if ticks and ticks[-1] < y_max:
        ticks.append(y_max)
    return ticks

# ================================================================
#                     Original single‑plot generators
# ================================================================

def gen_main_page(group_name: str, mode: str, *, y_min: float, y_max: float):
    assert mode in {"range", "tag"}
    plt.style.use(CONFIG["STYLE"]["matplotlib_style"])
    fig, axes = create_panel_layout(mosaic=[["A"]], figsize=(5, 3))

    algo_keys = CONFIG["ORDERS"][mode]
    (
        lines,
        labels,
        colors,
        markers,
        zorders,
        markersizes,
        line_widths,
        markeredgewidths,
    ) = _load_lines_for_group(group_name, algo_keys)

    # Matplotlib expects scalar linewidth/markeredgewidth per call; draw_free_lines forwards kwargs
    # Use per-config default; if all line widths (or mew) are equal, adopt that value
    scalar_lw = CONFIG["PLOTS"].get("default_line_width", 3)
    if line_widths:
        try:
            if len(set(line_widths)) == 1:
                scalar_lw = float(line_widths[0])
        except Exception:
            pass
    scalar_mew = CONFIG["PLOTS"].get("default_markeredgewidth", 1.0)
    if markeredgewidths:
        try:
            if len(set(markeredgewidths)) == 1:
                scalar_mew = float(markeredgewidths[0])
        except Exception:
            pass

    draw_free_lines(
        lines=lines,
        labels=labels,
        zorders=zorders,
        colors=colors,
        markers=markers,
        x_range=(80, 100),
        y_range=(y_min, y_max),
        log_scale=CONFIG["PLOTS"].get("use_log_y", True),
        linestyle=[CONFIG["PLOTS"].get("line_style", "-") for _ in lines],
        line_width=scalar_lw,
        markeredgewidth=scalar_mew,
        markersizes=markersizes,
        legend=False,
        ax=axes["A"],
        save=False,
    )

    out = Path(CONFIG["PATHS"]["FIG_ROOT"]) / f"{group_name}.svg"
    save_fig(fig, save_path=str(out))


# def gen_main_page_range(group_name: str, y_min=60, y_max=20000):
#     return gen_main_page(group_name, "range", y_min=y_min, y_max=y_max)


# def gen_main_page_tag(group_name: str, y_min=60, y_max=20000):
#     return gen_main_page(group_name, "tag", y_min=y_min, y_max=y_max)

# ================================================================
#         NEW: composite generators with shared Y ticks across 3 subplots
# ================================================================

def gen_composite(dataset: str, mode: str, *, y_min: float, y_max: float, figsize=(24, 5)):
    """
    Combine 4p/10p/40p into a single wide figure (A|B|C), sharing the same Y range & ticks.
    Only the left subplot shows Y tick labels (grid remains on others).
    """
    assert mode in {"range", "tag"}

    plt.style.use(CONFIG["STYLE"]["matplotlib_style"])
    fig, axes = create_panel_layout(mosaic=[["A", "B", "C"]], figsize=figsize)

    groups = [f"{dataset}-{mode}-4p", f"{dataset}-{mode}-10p", f"{dataset}-{mode}-40p"]
    ax_keys = ["A", "B", "C"]
    titles  = ["4%", "10%", "40%"]

    yticks = _nice_log_ticks(y_min, y_max, multipliers=CONFIG["PLOTS"].get("ytick_multipliers", (1,)))
    algo_keys = CONFIG["ORDERS"][mode]

    for idx, (g, k, ttl) in enumerate(zip(groups, ax_keys, titles)):
        (
            lines,
            labels,
            colors,
            markers,
            zorders,
            markersizes,
            line_widths,
            markeredgewidths,
        ) = _load_lines_for_group(g, algo_keys)

        scalar_lw = CONFIG["PLOTS"].get("default_line_width", 3)
        if line_widths:
            try:
                if len(set(line_widths)) == 1:
                    scalar_lw = float(line_widths[0])
            except Exception:
                pass
        scalar_mew = CONFIG["PLOTS"].get("default_markeredgewidth", 1.0)
        if markeredgewidths:
            try:
                if len(set(markeredgewidths)) == 1:
                    scalar_mew = float(markeredgewidths[0])
            except Exception:
                pass

        draw_free_lines(
            lines=lines,
            labels=labels,
            zorders=zorders,
            colors=colors,
            markers=markers,
            x_range=CONFIG["PLOTS"].get("x_range", (0.79, 1.01)),
            x_ticks=CONFIG["PLOTS"].get("x_ticks", [0.8, 0.85, 0.9, 0.95, 1.0]),
            y_range=(y_min, y_max),
            log_scale=CONFIG["PLOTS"].get("use_log_y", True),
            linestyle=[CONFIG["PLOTS"].get("line_style", "-") for _ in lines],
            line_width=scalar_lw,
            markeredgewidth=scalar_mew,
            markersizes=markersizes,
            legend=False,
            ax=axes[k],
            save=False,
        )

        axes[k].set_title(ttl)
        axes[k].set_yticks(yticks)
        if idx == 0:
            axes[k].set_ylabel("QPS (/s)", fontdict={'size': 22})
        if idx == 1:
            axes[k].set_xlabel("Recall Rate", fontdict={'size': 22})
        if idx > 0:
            axes[k].tick_params(labelleft=False)

    out_path = Path(CONFIG["PATHS"]["FIG_ROOT"]) / f"{dataset}-{mode}.svg"
    save_fig(fig, save_path=str(out_path))

# ================================================================
#                              Legend
# ================================================================

def gen_legend():
    from figure_tools_pro import save_legend_only_from_styles

    order = CONFIG["ORDERS"].get("legend")
    if not order:
        # union of range & tag while preserving order
        order = []
        for k in CONFIG["ORDERS"]["range"] + CONFIG["ORDERS"]["tag"]:
            if k not in order:
                order.append(k)

    labels, colors, markers, msz = [], [], [], []
    for key in order:
        spec = CONFIG["ALGORITHMS"].get(key)
        if not spec or not spec.get("visible", True):
            continue
        labels.append(spec.get("label", key))
        colors.append(spec.get("color"))
        markers.append(spec.get("marker", "o"))
        msz.append(spec.get("markersize", 7))

    plt.style.use(CONFIG["STYLE"]["matplotlib_style"])
    legend_cfg = CONFIG["PLOTS"]["legend"]

    # --- normalize markersize from config (avoid None) ---
    cfg_ms = legend_cfg.get("markersize", None)
    if cfg_ms is None:
        msizes = msz  # use per‑algorithm sizes collected above
    elif isinstance(cfg_ms, (int, float)):
        msizes = [float(cfg_ms)] * len(labels)
    else:
        # treat as iterable; pad/truncate to match number of labels
        try:
            msizes = list(cfg_ms)
        except Exception:
            msizes = msz
        if len(msizes) < len(labels):
            msizes = msizes + ([msz[0]] * (len(labels) - len(msizes)) if msz else [7] * (len(labels) - len(msizes)))
        msizes = msizes[:len(labels)]

    save_legend_only_from_styles(
        labels,
        colors,
        markers,
        str(Path(CONFIG["PATHS"]["FIG_ROOT"]) / legend_cfg.get("outfile", "legend.svg")),
        ncol=legend_cfg.get("ncol", 10),
        linewidth=legend_cfg.get("linewidth", 2),
        markersize=msizes,  # always a concrete list of floats
        markeredgewidth=legend_cfg.get("markeredgewidth", 0.5),
        columnspacing=legend_cfg.get("columnspacing", 2.0),
    )

# ================================================================
#                              CLI
# ================================================================
if __name__ == "__main__":
    # Optional: output legend first
    gen_legend()

    ds_settings = CONFIG["PLOTS"]["dataset_settings"]
    # for dataset, modes in ds_settings.items():
    #     for mode, (y_min, y_max) in modes.items():
    #         gen_composite(
    #             dataset,
    #             mode,
    #             y_min=y_min,
    #             y_max=y_max,
    #             figsize=CONFIG["PLOTS"].get("figsize_composite", (16, 4)),
    #         )

    # Example: single figures (kept for compatibility)
    groups = [
        "RANDOM-d128-n50000",
        "RANDOM-d128-n100000",
    ]
    for g in groups:
        mode = "range" if "-range-" in g else "tag"
        ds = g.split("-")[0]
        y_min, y_max = ds_settings[ds][mode]
        gen_main_page(g, mode, y_min=y_min, y_max=y_max)
