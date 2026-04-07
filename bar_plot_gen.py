"""
Generate grouped bar charts (histogram-like) using figure_tools_pro.draw_grouped_bar.

Fixed 3 settings (hard-coded by default):
- SIFT1M-0.00025
- SIFT1M-0.00027
- SIFT1M-0.0003

Data format (JSON) if you still want to pass --input:
{
  "group_labels": ["16","24","32"],
  "x_label": "$M$",
  "y_label": "Index Size (MB)",
  "settings": {
    "SIFT1M-0.00025": {
      "series": [
        {"label":"DAL","values":[...], "edgecolor":"#ff7f0e","hatch":"xxx","facecolor":"white"},
        {"label":"DLH/DLH-M","values":[...],        "edgecolor":"#d62728","hatch":"xxx","facecolor":"white"},
        {"label":"HNSW","values":[...],       "edgecolor":"#1f77b4","hatch":"xxx","facecolor":"white"}
      ]
    },
    "SIFT1M-0.00027": { "series": [ ... ] },
    "SIFT1M-0.0003":  { "series": [ ... ] }
  }
}
"""

import argparse
import json
import os
import re
import math  # ✅ add
from pathlib import Path
from typing import Any, Dict, List, Tuple  # ✅ add Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # for legend proxies

os.chdir(os.path.dirname(__file__))

from figure_tools_pro import draw_grouped_bar, save_fig


SETTING_NAMES = ["SIFT1M-0.00025", "SIFT1M-0.00027", "SIFT1M-0.0003"]

DEFAULT_EXAMPLE: Dict[str, Any] = {
    "group_labels": ["16", "24", "32"],
    "x_label": r"$M$",
    "y_label": "Index Size (MB)",
    "settings": {
        # 下面是样例数据：每个 setting 下，每个算法 3 个值（对应 M=16/24/32）
        # HNSW 为 hnsw_size + graph_size 
        # 我们的方法为 hnsw_size + label_size
        "SIFT1M-0.00025": {
            "series": [
                {"label": "DAL", "values": [(630 + 500), (691 + 500), (752 + 500)], "edgecolor": "#ff7f0e", "hatch": "xx", "facecolor": "white"},
                {"label": "DLH/DLH-M",        "values": [(630 + 218),  (691 + 218),  (752 + 218) ], "edgecolor": "#d62728", "hatch": "xx", "facecolor": "white"},
                {"label": "HNSW",       "values": [(630 + 10),  (691 + 10),  (752 + 10) ], "edgecolor": "#1f77b4", "hatch": "xx", "facecolor": "white"},
            ]
        },
        "SIFT1M-0.00027": {
            "series": [
                {"label": "DAL", "values": [(630 + 586), (691 + 586), (752 + 586)], "edgecolor": "#ff7f0e", "hatch": "xx", "facecolor": "white"},
                {"label": "DLH/DLH-M",        "values": [(630 + 247),  (691 + 247),  (752 + 247) ], "edgecolor": "#d62728", "hatch": "xx", "facecolor": "white"},
                {"label": "HNSW",       "values": [(630 + 10.5),  (691 + 10.5),  (752 + 10.5) ], "edgecolor": "#1f77b4", "hatch": "xx", "facecolor": "white"},
            ]
        },
        "SIFT1M-0.0003": {
            "series": [
                {"label": "DAL", "values": [(630 + 678), (691 + 678), (752 + 678)], "edgecolor": "#ff7f0e", "hatch": "xx", "facecolor": "white"},
                {"label": "DLH/DLH-M",        "values": [(630 + 296),  (691 + 296), (752 + 296)], "edgecolor": "#d62728", "hatch": "xx", "facecolor": "white"},
                {"label": "HNSW",       "values": [(630 + 11),  (691 + 11),  (752 + 11) ], "edgecolor": "#1f77b4", "hatch": "xx", "facecolor": "white"},
            ]
        },
    },
}

PLOT_STYLE = "ggplot"

# 输出目录：所有文件放在同一目录（不建子文件夹）
OUT_DIR = Path("figures") / "index-size"
LEGEND_PATH = OUT_DIR / "legend.svg"

# 字体大小（贴近你给的示例）
FONT_AXIS = 12
FONT_TICK = 10
FONT_LEGEND = 11

# ✅ 纵轴刻度风格：尽量像示例图那样“少而整齐”
Y_TICK_TARGET = 4          # 目标主刻度数量（类似 0,5,10,15 -> 4 个）
Y_TICK_MAX = 6             # 最多主刻度数量
Y_HEADROOM_RATIO = 0.02    # 顶部留白比例，避免最高柱贴顶

# ✅ legend 导出裁剪留白（单位：英寸）
LEGEND_PAD_INCH = 0.02


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s or "setting"


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # 固定 3 setting
    if "group_labels" not in payload or "settings" not in payload:
        raise ValueError("JSON 必须包含 group_labels 和 settings 字段")

    if not isinstance(payload["group_labels"], list) or len(payload["group_labels"]) == 0:
        raise ValueError("group_labels 必须是非空数组")

    settings = payload["settings"]
    if not isinstance(settings, dict):
        raise ValueError("settings 必须是对象（dict）")

    # 强制三个 setting 都存在
    for name in SETTING_NAMES:
        if name not in settings:
            raise ValueError(f"settings 缺少固定 setting: {name}")
        sp = settings[name]
        if not isinstance(sp, dict) or "series" not in sp:
            raise ValueError(f"settings['{name}'] 必须是对象且包含 series")
        if not isinstance(sp["series"], list) or len(sp["series"]) == 0:
            raise ValueError(f"settings['{name}'].series 必须是非空数组")

        n_groups = len(payload["group_labels"])
        for s in sp["series"]:
            if "label" not in s or "values" not in s:
                raise ValueError(f"settings['{name}'].series 每个元素必须包含 label 和 values")
            if len(s["values"]) != n_groups:
                raise ValueError(f"settings['{name}'] 里 series '{s.get('label')}' 的 values 长度必须等于 group_labels 长度")

    return payload


def _force_white_facecolors(series: List[Dict[str, Any]]) -> List[str]:
    # 强制白色填充，避免 hatch 背景透明导致“漏底色/网格线”
    inner_colors: List[str] = []
    for s in series:
        fc = str(s.get("facecolor", "white")).strip().lower()
        if fc in {"none", "transparent", ""}:
            fc = "white"
        inner_colors.append(fc)
    return inner_colors


def _flatten_all_values(payload: Dict[str, Any]) -> List[float]:
    vals: List[float] = []
    for setting_name in SETTING_NAMES:
        series = payload["settings"][setting_name]["series"]
        for s in series:
            vals.extend([float(x) for x in s["values"]])
    return vals


def _nice_step(raw: float) -> float:
    """Return a 'nice' step size >= raw using 1/2/5 * 10^k."""
    if raw <= 0:
        return 1.0
    k = math.floor(math.log10(raw))
    base = 10 ** k
    for m in (1, 2, 5, 10):
        step = m * base
        if step >= raw:
            return float(step)
    return float(10 * base)


def _compute_fixed_y_axis(payload: Dict[str, Any]) -> Tuple[float, List[float]]:
    """Compute a shared y_max and y_ticks for all settings based on values."""
    vals = _flatten_all_values(payload)
    vmax = max(vals) if vals else 1.0
    vmax = vmax * (1.0 + Y_HEADROOM_RATIO)

    # 目标：大概 Y_TICK_TARGET 个主刻度 => 间隔约 vmax/(Y_TICK_TARGET-1)
    raw_step = vmax / max(1, (Y_TICK_TARGET - 1))
    step = _nice_step(raw_step)

    y_max = math.ceil(vmax / step) * step
    ticks = [i * step for i in range(0, int(round(y_max / step)) + 1)]

    # 若刻度太密，逐步放大 step（保持“像示例图一样简洁”）
    while len(ticks) > Y_TICK_MAX:
        step = step * 2
        y_max = math.ceil(vmax / step) * step
        ticks = [i * step for i in range(0, int(round(y_max / step)) + 1)]

    return float(y_max), [float(t) for t in ticks]


def plot_grouped_bar_for_setting(
    base_payload: Dict[str, Any],
    setting_name: str,
    out_path: Path,
    *,
    figsize=(6.0, 3.2),
    y_max: float | None = None,
    y_ticks: List[float] | None = None,
) -> str:
    base_payload = normalize_payload(base_payload)

    setting = base_payload["settings"][setting_name]
    series = setting["series"]

    group_labels: List[str] = [str(x) for x in base_payload["group_labels"]]
    serie_labels: List[str] = [str(s["label"]) for s in series]
    data_series: List[List[float]] = [list(map(float, s["values"])) for s in series]

    colors: List[str] = [str(s.get("edgecolor", "")) for s in series]
    hatches: List[str] = [str(s.get("hatch", "")) for s in series]
    inner_colors: List[str] = _force_white_facecolors(series)

    plt.style.use(PLOT_STYLE)

    ax = draw_grouped_bar(
        data_series=data_series,
        serie_labels=serie_labels,
        group_labels=group_labels,
        x_label="",
        y_label="",
        pic_width=float(figsize[0]),
        pic_height=float(figsize[1]),
        colors=colors if all(colors) else [],
        hatches=hatches if all(hatches) else [],
        inner_colors=inner_colors,
        legend=False,
        save=False,
        return_ax=True,
    )

    # ✅ 固定纵轴划分（统一 y-lim / y-ticks，使得观感更像你贴图）
    if y_max is not None:
        ax.set_ylim(0.0, float(y_max))
    if y_ticks is not None and len(y_ticks) > 0:
        ax.set_yticks([float(t) for t in y_ticks])

    # 统一字体
    x_label = base_payload.get("x_label", "")
    y_label = base_payload.get("y_label", "")
    if x_label:
        ax.set_xlabel(x_label, fontsize=FONT_AXIS)
    if y_label:
        ax.set_ylabel(y_label, fontsize=FONT_AXIS)
    ax.tick_params(axis="both", which="both", labelsize=FONT_TICK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return save_fig(ax.figure, save_path=str(out_path))


def save_shared_legend(base_payload: Dict[str, Any], legend_path: Path, *, legend_setting: str = "SIFT1M-0.00025") -> str:
    base_payload = normalize_payload(base_payload)

    series = base_payload["settings"][legend_setting]["series"]
    plt.style.use(PLOT_STYLE)

    handles: List[Patch] = []
    labels: List[str] = []

    for s in series:
        label = str(s.get("label", ""))
        edgecolor = str(s.get("edgecolor", "#000000"))
        hatch = str(s.get("hatch", ""))
        fc = str(s.get("facecolor", "white")).strip().lower()
        if fc in {"none", "transparent", ""}:
            fc = "white"

        handles.append(Patch(facecolor=fc, edgecolor=edgecolor, hatch=hatch, linewidth=1.5))
        labels.append(label)

    # 仍然给一个初始 figsize，但最终会按 legend bbox 裁剪输出
    fig = plt.figure(figsize=(8.0, 0.9))
    ax = fig.add_subplot(111)
    ax.axis("off")

    leg = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=True,
        fontsize=FONT_LEGEND,
        handlelength=2.0,
        handleheight=1.0,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    legend_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ 关键：按 legend 的真实边界框裁剪，尽量减少四周白边
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = leg.get_window_extent(renderer=renderer).transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1.02, 1.20)  # 稍微放大一点，避免边框被裁掉

    fig.savefig(
        legend_path,
        format="svg",
        bbox_inches=bbox,
        pad_inches=LEGEND_PAD_INCH,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    return str(legend_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", type=str, default="", help="可选：传入多-setting JSON；不提供则用 DEFAULT_EXAMPLE")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="输出目录（默认 ./figures/index-size）")
    args = ap.parse_args()

    payload = DEFAULT_EXAMPLE if not args.input else load_json(args.input)
    payload = normalize_payload(payload)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 计算全局统一的纵轴范围与刻度
    y_max, y_ticks = _compute_fixed_y_axis(payload)

    # shared legend (only once)
    legend_path = out_dir / "legend.svg"
    save_shared_legend(payload, legend_path, legend_setting=SETTING_NAMES[0])
    print(f"Saved shared legend: {legend_path}")

    # one plot per setting
    for setting_name in SETTING_NAMES:
        fname = f"{_sanitize_filename(setting_name)}-index-size.svg"
        out_path = out_dir / fname
        plot_grouped_bar_for_setting(payload, setting_name, out_path, y_max=y_max, y_ticks=y_ticks)
        print(f"Saved plot [{setting_name}]: {out_path}")


if __name__ == "__main__":
    main()