"""
Refactored plotting library for custom layouts and unified parameters.

This module provides a suite of functions for creating various charts—line plots,
flagged line plots with class-based highlighting, grouped bar charts, breakdown
horizontal bar charts, and multi-index scatter plots with optional trend lines.

Key features:

* Unified parameter names for better consistency across functions.
* Optional ``ax`` argument in all plotting functions for embedding plots
  into custom subplot layouts. When no ``ax`` is provided, a new figure
  is created automatically.
* Support for custom named mosaic layouts via ``create_panel_layout``.
* Helper functions for legends, axis formatting, and figure saving to
  reduce duplication.
* Backwards-compatible defaults matching the original functions’ behaviour.
* NEW: Optional zorder controls for each plotting function (defaults preserve
  original stacking behaviour).

Usage examples are provided in the accompanying README or project documentation.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Dict, Any

from pathlib import Path

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _legend(ax: matplotlib.axes.Axes,
            legend: bool,
            legend_loc: Union[str, Tuple[float, float, float, float]] = 'upper left',
            n_cols: int = 1) -> None:
    """Draw a legend on ``ax`` using a flexible location specification.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the legend.
    legend : bool
        Whether to display a legend.
    legend_loc : str or tuple
        If a string, it is forwarded directly to ``ax.legend(loc=...)``.
        If a tuple, its interpretation depends on its length:

        * 2 values: interpreted as a ``bbox_to_anchor`` point; legend is placed
          at 'upper left'.
        * 3 values: (x, y, loc) where (x, y) is ``bbox_to_anchor`` and ``loc``
          is the legend location code (e.g. 'upper left').
        * 4 values: (x, y, loc, pad) to specify ``borderaxespad``.
    n_cols : int
        Number of columns in the legend.
    """
    if not legend:
        return
    if isinstance(legend_loc, str):
        ax.legend(loc=legend_loc, ncol=n_cols, frameon=True)
    elif isinstance(legend_loc, tuple):
        if len(legend_loc) == 4:
            x, y, loc, pad = legend_loc
            ax.legend(bbox_to_anchor=(x, y), loc=loc, borderaxespad=pad,
                      ncol=n_cols, frameon=True)
        elif len(legend_loc) == 3:
            x, y, loc = legend_loc
            ax.legend(bbox_to_anchor=(x, y), loc=loc, ncol=n_cols,
                      frameon=True)
        elif len(legend_loc) == 2:
            x, y = legend_loc
            ax.legend(bbox_to_anchor=(x, y), loc='upper left', ncol=n_cols,
                      frameon=True)
        else:
            ax.legend(ncol=n_cols, frameon=True)
    else:
        ax.legend(ncol=n_cols, frameon=True)


def _apply_axes_format(ax: matplotlib.axes.Axes,
                        x_label: str = "",
                        y_label: str = "",
                        x_range: Tuple[float, float] = (),
                        y_range: Tuple[float, float] = (),
                        x_ticks: Optional[List[float]] = None,
                        y_ticks: Optional[List[float]] = None,
                        x_tick_labels: Optional[List[str]] = None,
                        x_log_scale: bool = False,
                        y_log_scale: bool = False,
                        tick_labelsize: int = 20) -> None:
    """Apply common formatting to axes.

    Sets labels, ranges, tick positions, tick labels, and log scales if
    requested. If a label is empty, it will not modify the corresponding
    axis title. Tick label sizes can be customised with ``tick_labelsize``.
    """
    if x_label:
        ax.set_xlabel(x_label, fontdict={'size': 22})
    if y_label:
        ax.set_ylabel(y_label, fontdict={'size': 22})
    ax.tick_params(labelsize=tick_labelsize)
    if x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    if x_ticks is not None:
        if x_tick_labels is not None and len(x_tick_labels) == len(x_ticks):
            ax.set_xticks(x_ticks, x_tick_labels)
        else:
            ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if x_log_scale:
        ax.set_xscale('log')
    if y_log_scale:
        ax.set_yscale('log')


def _ensure_figure_ax(ax: Optional[matplotlib.axes.Axes],
                      figsize: Tuple[float, float],
                      style: Optional[str] = "ggplot") -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, bool]:
    """Return a figure and axis, creating them if necessary.

    If ``ax`` is ``None``, a new figure and axes are created with the given
    ``figsize`` and optional ``style`` applied. The return value includes a
    boolean indicating whether a new figure was created.
    """
    created = False
    if ax is None:
        if style:
            plt.style.use(style)
        fig, ax = plt.subplots(figsize=figsize)
        created = True
    else:
        fig = ax.figure
    return fig, ax, created


def create_panel_layout(
    mosaic: List[List[str]],
    figsize: Tuple[float, float] = (10, 6),
    width_ratios: Optional[List[float]] = None,
    height_ratios: Optional[List[float]] = None,
    constrained: bool = True
) -> Tuple[matplotlib.figure.Figure, Dict[str, matplotlib.axes.Axes]]:
    """Create a named mosaic of subplots for flexible panel layouts.

    Parameters
    ----------
    mosaic : list of list of str
        A 2D list defining the grid; identical strings are merged into a
        single subplot spanning their region. Distinct strings define
        separate axes accessible via the returned dictionary.
    figsize : tuple
        Size of the entire figure in inches.
    width_ratios : list, optional
        Relative widths of each column. Length must match the number of
        columns in the mosaic.
    height_ratios : list, optional
        Relative heights of each row. Length must match the number of
        rows in the mosaic.
    constrained : bool
        Use Matplotlib's ``constrained_layout`` if True to automatically
        adjust subplot spacing. If False, you may need to call
        ``fig.tight_layout()`` manually later.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : dict
        A dictionary mapping the mosaic keys to corresponding axes.
    """
    gridspec_kw: Dict[str, Any] = {}
    if width_ratios is not None:
        gridspec_kw['width_ratios'] = width_ratios
    if height_ratios is not None:
        gridspec_kw['height_ratios'] = height_ratios
    fig_kwargs: Dict[str, Any] = {"figsize": figsize}
    if constrained:
        fig_kwargs["layout"] = "constrained"
    fig, axes = plt.subplot_mosaic(mosaic, gridspec_kw=gridspec_kw, **fig_kwargs)
    return fig, axes


def save_fig(fig: matplotlib.figure.Figure,
             title: str = "",
             save_path: Optional[str] = None,
             dpi: int = 300) -> str:
    """Save a figure to disk, returning the output filename.

    If ``save_path`` has an extension (e.g., .pdf/.svg/.png), the format is
    chosen accordingly. If no extension is given, .png is used by default.
    When ``save_path`` is None, the filename is derived from ``title`` (or
    'figure.png') in the current working directory. The figure is saved with
    ``bbox_inches='tight'`` to minimize whitespace.
    """
    if save_path:
        p = Path(save_path)
        ext = p.suffix.lower()
        if not ext:                       # no extension -> default to .png
            p = p.with_suffix(".png")
            ext = ".png"

        fmt = ext.lstrip(".")             # e.g. ".pdf" -> "pdf"
        # allow common formats; fall back to png if unknown
        allowed = {"png", "pdf", "svg", "jpg", "jpeg", "tif", "tiff", "eps"}
        if fmt not in allowed:
            p = p.with_suffix(".png")
            fmt = "png"

        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight", format=fmt)
        return str(p)
    else:
        out = f"{title}.png" if title else "figure.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        return out


# -----------------------------------------------------------------------------
# Main plotting functions
# -----------------------------------------------------------------------------

def draw_free_lines(
        lines: List[List[Tuple[float, float]]],
        labels: List[str],
        *,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        pic_width: float = 8,
        pic_height: float = 5,
        show: bool = False,
        x_range: Tuple[float, float] = (),
        x_ticks: List[float] = [],
        x_tick_labels: List[str] = [],
        y_range: Tuple[float, float] = (),
        y_ticks: List[float] = [],
        log_scale: bool = False,
        x_log_scale: bool = False,
        legend: bool = False,
        legend_loc: Union[str, Tuple[float, float, float, float]] = 'upper left',
        n_cols: int = 1,
        line_width: float = 3.5,
        linestyle: List[str] = [],
        colors: List[str] = [],
        markers: List[str] = [],
        markersizes: List[float] = [],
        markeredgewidth: float = 2.5,
        # --- NEW ---
        zorders: List[Optional[int]] = [],
        # -----------
        ax: Optional[matplotlib.axes.Axes] = None,
        style: Optional[str] = "ggplot",
        save: bool = True,
        save_path: Optional[str] = None,
        return_ax: bool = False
    ) -> Optional[matplotlib.axes.Axes]:
    """Plot multiple freeform line series on a single axis or a supplied axis.

    ``zorders`` (optional) lets you set per-series zorder without changing
    defaults; if an entry is None or missing, the Matplotlib default is used.
    """
    fig, ax, created = _ensure_figure_ax(ax, (pic_width, pic_height), style=style)
    # Plot each line
    for i, line in enumerate(lines):
        xs = [x for x, _ in line]
        ys = [y for _, y in line]
        c  = colors[i]      if i < len(colors)      else None
        m  = markers[i]     if i < len(markers)     else None
        ms = markersizes[i] if i < len(markersizes) else None
        ls = linestyle[i]   if i < len(linestyle)   else None
        kwargs: Dict[str, Any] = dict(
            label=(labels[i] if i < len(labels) else None),
            marker=m, ms=ms, linestyle=ls,
            mfc=c, mew=markeredgewidth, lw=line_width, color=c
        )
        if i < len(zorders) and zorders[i] is not None:
            kwargs["zorder"] = zorders[i]
        ax.plot(xs, ys, **kwargs)
    # Format axes and legend
    _apply_axes_format(ax,
                       x_label=x_label,
                       y_label=y_label,
                       x_range=x_range,
                       y_range=y_range,
                       x_ticks=x_ticks if x_ticks else None,
                       y_ticks=y_ticks if y_ticks else None,
                       x_tick_labels=x_tick_labels if x_tick_labels else None,
                       x_log_scale=x_log_scale,
                       y_log_scale=log_scale)
    _legend(ax, legend, legend_loc, n_cols)
    if created:
        fig.tight_layout()
        if save:
            save_fig(fig, title=title, save_path=save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return ax if return_ax else None


def draw_free_lines_extended(
        lines: List[List[Tuple[float, float, Any]]],
        labels: List[str],
        *,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        pic_width: float = 8,
        pic_height: float = 5,
        show: bool = False,
        x_range: Tuple[float, float] = (),
        x_ticks: List[float] = [],
        x_tick_labels: List[str] = [],
        y_range: Tuple[float, float] = (),
        y_ticks: List[float] = [],
        log_scale: bool = False,
        x_log_scale: bool = False,
        legend: bool = False,
        legend_loc: Union[str, Tuple[float, float, float, float]] = 'upper left',
        n_cols: int = 1,
        line_width: float = 3.5,
        linestyle: List[str] = [],
        colors: List[str] = [],
        markers: List[str] = [],
        markersizes: List[float] = [],
        markeredgewidth: float = 2.5,
        highlight_true: bool = True,
        true_colors: Union[List[str], str] = [],
        true_markers: Union[List[str], str] = [],
        true_markersizes: Union[List[float], float] = [],
        true_alpha: float = 1.0,
        true_zorder: int = 3,
        true_edgecolor: Optional[str] = None,
        enable_flag_classes: bool = True,
        flag_style_map: Optional[Dict[Any, Dict[str, Any]]] = None,
        highlight_flags: Optional[Union[set, list]] = None,
        flag_none: Any = None,
        class_default_color: Optional[str] = None,
        class_default_marker: Optional[str] = None,
        class_default_size: Optional[float] = None,
        class_default_alpha: Optional[float] = None,
        class_default_edgecolor: Optional[str] = None,
        class_default_zorder: int = 3,
        # --- NEW ---
        line_zorders: List[Optional[int]] = [],
        # -----------
        ax: Optional[matplotlib.axes.Axes] = None,
        style: Optional[str] = "ggplot",
        save: bool = True,
        save_path: Optional[str] = None,
        return_ax: bool = False
    ) -> Optional[matplotlib.axes.Axes]:
    """Plot line series with optional boolean and multi-class highlighting.

    ``line_zorders`` (optional) sets per-series zorder for the base lines.
    Boolean (`true_zorder`) and class highlights already support zorder via
    parameters and ``flag_style_map``; defaults preserved.
    """
    fig, ax, created = _ensure_figure_ax(ax, (pic_width, pic_height), style=style)
    for i, line in enumerate(lines):
        xs: List[float] = []
        ys: List[float] = []
        flags: List[Any] = []
        has_any_flag = False
        # unpack points
        for p in line:
            if len(p) == 2:
                x, y = p  # type: ignore
                xs.append(float(x))
                ys.append(float(y))
                flags.append(flag_none)
            elif len(p) == 3:
                x, y, f = p  # type: ignore
                xs.append(float(x))
                ys.append(float(y))
                flags.append(f)
                has_any_flag = True
            else:
                raise ValueError("Each point must be (x,y) or (x,y,flag).")
        c  = colors[i]      if i < len(colors)      else None
        m  = markers[i]     if i < len(markers)     else None
        ms = markersizes[i] if i < len(markersizes) else None
        ls = linestyle[i]   if i < len(linestyle)   else '-'
        kwargs: Dict[str, Any] = dict(
            label=(labels[i] if i < len(labels) else None),
            marker=m, ms=ms, linestyle=ls,
            mfc=c, mew=markeredgewidth, lw=line_width, color=c
        )
        if i < len(line_zorders) and line_zorders[i] is not None:
            kwargs["zorder"] = line_zorders[i]
        ax.plot(xs, ys, **kwargs)
        # Highlight boolean True
        if highlight_true and has_any_flag and any(isinstance(f, bool) and f is True for f in flags):
            idx_true = [j for j, f in enumerate(flags) if isinstance(f, bool) and f is True]
            if idx_true:
                x_true = [xs[j] for j in idx_true]
                y_true = [ys[j] for j in idx_true]
                # choose styles for boolean True
                if isinstance(true_colors, list) and len(true_colors) > i:
                    t_color = true_colors[i]
                elif isinstance(true_colors, str) and true_colors:
                    t_color = true_colors
                else:
                    t_color = 'tab:red'
                if isinstance(true_markers, list) and len(true_markers) > i:
                    t_marker = true_markers[i]
                elif isinstance(true_markers, str) and true_markers:
                    t_marker = true_markers
                else:
                    t_marker = m if m is not None else 'o'
                if isinstance(true_markersizes, list) and len(true_markersizes) > i:
                    t_ms = true_markersizes[i]
                elif isinstance(true_markersizes, (int, float)) and true_markersizes:
                    t_ms = true_markersizes  # type: ignore
                else:
                    t_ms = (ms if ms is not None else 6) * 1.15
                ax.scatter(x_true, y_true,
                           marker=t_marker,
                           s=(t_ms ** 2) * 0.8,
                           facecolors=t_color,
                           edgecolors=(true_edgecolor if true_edgecolor is not None else t_color),
                           linewidths=markeredgewidth,
                           alpha=true_alpha,
                           zorder=true_zorder,
                           label=None)
        # Highlight categorical flags
        if enable_flag_classes and flag_style_map:
            target_flags: set[Any] = set(highlight_flags) if highlight_flags is not None else set(flag_style_map.keys())
            group: Dict[Any, List[int]] = {}
            for j, f in enumerate(flags):
                if f == flag_none:
                    continue
                if f in target_flags:
                    group.setdefault(f, []).append(j)
            for fval, idxs in group.items():
                if not idxs:
                    continue
                x_f = [xs[j] for j in idxs]
                y_f = [ys[j] for j in idxs]
                st = flag_style_map.get(fval, {})
                t_color = st.get('color',
                                 class_default_color if class_default_color is not None else (c if c is not None else 'tab:red'))
                t_marker = st.get('marker',
                                  class_default_marker if class_default_marker is not None else (m if m is not None else 'o'))
                ms_val = st.get('size',
                                class_default_size if class_default_size is not None else (ms if ms is not None else 6) * 1.15)
                t_alpha = st.get('alpha',
                                 class_default_alpha if class_default_alpha is not None else 1.0)
                t_ec = st.get('edgecolor',
                              class_default_edgecolor if class_default_edgecolor is not None else t_color)
                t_z = st.get('zorder', class_default_zorder if class_default_zorder is not None else 3)
                ax.scatter(x_f, y_f,
                           marker=t_marker,
                           s=(ms_val ** 2) * 0.8,
                           facecolors=t_color,
                           edgecolors=t_ec,
                           linewidths=markeredgewidth,
                           alpha=t_alpha,
                           zorder=t_z,
                           label=None)
    # Format axes and legend
    _apply_axes_format(ax,
                       x_label=x_label,
                       y_label=y_label,
                       x_range=x_range,
                       y_range=y_range,
                       x_ticks=x_ticks if isinstance(x_ticks, list) else None,
                       y_ticks=y_ticks if isinstance(x_ticks, list) else None,
                       x_tick_labels=x_tick_labels if x_tick_labels else None,
                       x_log_scale=x_log_scale,
                       y_log_scale=log_scale)
    _legend(ax, legend, legend_loc, n_cols)
    if created:
        fig.tight_layout()
        if save:
            save_fig(fig, title=title, save_path=save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return ax if return_ax else None


def draw_grouped_bar(
        data_series: List[List[float]],
        serie_labels: List[str],
        group_labels: List[str],
        *,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        pic_width: float = 8,
        pic_height: float = 5,
        show: bool = False,
        y_range: Tuple[float, float] = (),
        y_ticks: List[float] = [],
        log_scale: bool = False,
        legend: bool = False,
        legend_loc: Union[str, Tuple[float, float, float, float]] = 'upper left',
        n_cols: int = 1,
        bar_width: float = 0.8,
        bar_padding: float = 0.2,
        group_padding: float = 1,
        end_padding: float = 1,
        colors: List[str] = [],
        hatches: List[str] = [],
        inner_colors: List[str] = [],
        # --- NEW ---
        bar_zorders: List[Optional[int]] = [],
        # -----------
        ax: Optional[matplotlib.axes.Axes] = None,
        style: Optional[str] = "ggplot",
        save: bool = True,
        save_path: Optional[str] = None,
        return_ax: bool = False
    ) -> Optional[matplotlib.axes.Axes]:
    """Create a grouped bar chart.

    ``bar_zorders`` (optional) sets per-series zorder for the bars. If an entry
    is None/missing, default stacking is preserved.
    """
    n_groups = len(group_labels)
    n_series = len(data_series)
    # compute width for each group
    group_width = n_series * bar_width + (n_series - 1) * bar_padding
    group_x_start = lambda i: end_padding + i * (group_width + group_padding)
    group_x_tick_pos = lambda i: group_x_start(i) + group_width / 2
    group_x_bar_pos = lambda i, j: group_x_start(i) + j * (bar_width + bar_padding) + bar_width / 2
    bar_x_poss_all = [[group_x_bar_pos(i, j) for i in range(n_groups)] for j in range(n_series)]
    tick_x_pos_all = [group_x_tick_pos(i) for i in range(n_groups)]
    fig, ax, created = _ensure_figure_ax(ax, (pic_width, pic_height), style=style)
    ax.tick_params(labelsize=20)
    # Plot bars
    for j in range(n_series):
        kwargs: Dict[str, Any] = dict(
            width=bar_width,
            label=serie_labels[j],
            edgecolor=None if len(colors) != n_series else colors[j],
            hatch=None if len(hatches) != n_series else hatches[j],
            color=None if len(inner_colors) != n_series else inner_colors[j],
            linewidth=2 if legend else 4
        )
        if j < len(bar_zorders) and bar_zorders[j] is not None:
            kwargs["zorder"] = bar_zorders[j]
        ax.bar(bar_x_poss_all[j], data_series[j], **kwargs)
    # X tick positions and labels
    ax.set_xticks(tick_x_pos_all, group_labels)
    # Axis labels
    if y_label:
        ax.set_ylabel(y_label, fontsize=20)
    if x_label:
        ax.set_xlabel(x_label, fontsize=20)
    # Y ticks and range
    if y_ticks:
        ax.set_yticks(y_ticks)
    if log_scale:
        ax.set_yscale('log')
    if y_range and len(y_range) == 2:
        ax.set_ylim(*y_range)
    # X range so bars are nicely spaced
    ax.set_xlim(0, group_x_start(n_groups - 1) + group_width + end_padding)
    _legend(ax, legend, legend_loc, n_cols)
    if created:
        fig.tight_layout()
        if save:
            save_fig(fig, title=title, save_path=save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return ax if return_ax else None


def draw_breakdown_bars(
    data: List[List[float]],
    labels: List[str],
    phase_labels: List[str],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    pic_width: float = 8,
    pic_height: float = 5,
    show: bool = False,
    x_range: Tuple[float, float] = (),
    x_ticks: Optional[List[float]] = None,
    y_range: Tuple[float, float] = (),
    y_ticks: Optional[List[float]] = None,
    log_scale: bool = False,
    legend: bool = False,
    legend_loc: Union[str, Tuple] = 'upper left',
    n_cols: int = 1,
    bar_width: float = 0.5,
    bar_margin: float = 0.3,
    colors: Optional[List[str]] = None,
    label_pct: bool = True,
    label_mode: str = "ratio",
    label_threshold: float = 0.1,
    label_fmt: str = "{:.0f}%",
    label_fontsize: int = 12,
    label_color: str = "white",
    label_inside: bool = True,
    label_outside_pad: float = 0.02,
    # --- NEW ---
    segment_zorders: Optional[List[Optional[int]]] = None,
    # -----------
    ax: Optional[matplotlib.axes.Axes] = None,
    style: Optional[str] = "ggplot",
    save: bool = True,
    save_path: Optional[str] = None,
    return_ax: bool = False
) -> Optional[matplotlib.axes.Axes]:
    """Draw horizontal stacked bars representing breakdown of phases per query.

    ``segment_zorders`` (optional) sets per-phase zorder for stacked segments
    (length == len(phase_labels)); None/missing entries preserve defaults.
    """
    num_queries = len(data)
    num_phases = len(phase_labels)
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = prop_cycle[:num_phases]
    y_positions = [i * (bar_width + bar_margin) for i in range(num_queries)]
    fig, ax, created = _ensure_figure_ax(ax, (pic_width, pic_height), style=style)
    ax.tick_params(labelsize=20)
    if log_scale:
        ax.set_xscale('log')
    for i, row in enumerate(data):
        start = 0.0
        total = sum(row) if sum(row) > 0 else 1e-12
        for j, duration in enumerate(row):
            kwargs: Dict[str, Any] = dict(
                y=y_positions[i], width=duration, left=start, height=bar_width,
                color=colors[j], label=phase_labels[j] if (legend and i == 0) else None
            )
            if segment_zorders is not None and j < len(segment_zorders) and segment_zorders[j] is not None:
                kwargs["zorder"] = segment_zorders[j]
            ax.barh(**kwargs)
            # Percent label
            if label_pct and duration > 0:
                ratio = duration / total
                cond = (label_mode == "ratio" and ratio >= label_threshold) or \
                       (label_mode == "absolute" and duration >= label_threshold)
                if cond:
                    text = label_fmt.format(ratio * 100.0)
                    if label_inside:
                        tx = start + duration / 2.0
                        ha = 'center'
                    else:
                        tx = start + duration * (1.0 + label_outside_pad)
                        ha = 'left'
                    ty = y_positions[i]
                    ax.text(tx, ty, text, ha=ha, va='center',
                            fontsize=label_fontsize, color=label_color)
            start += duration
    # Axis labels
    if x_label:
        ax.set_xlabel(x_label, fontsize=20)
    if y_label:
        ax.set_ylabel(y_label, fontsize=20)
    # Axis ranges and ticks
    if x_range:
        ax.set_xlim(x_range)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_range:
        ax.set_ylim(y_range)
    if y_ticks is not None:
        ax.set_yticks(y_positions, y_ticks)
    else:
        ax.set_yticks(y_positions, labels)
    _legend(ax, legend, legend_loc, n_cols)
    if created:
        fig.tight_layout()
        if save:
            save_fig(fig, title=title, save_path=save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return ax if return_ax else None


def draw_multi_index_scatter(
        series_list: List[List[int]],
        labels: List[str] = [],
        *,
        title: str = "",
        x_label: str = "i",
        y_label: str = "values[i]",
        pic_width: float = 8,
        pic_height: float = 5,
        show: bool = False,
        x_range: Tuple[float, float] = (),
        x_ticks: List[float] = [],
        y_range: Tuple[float, float] = (),
        y_ticks: List[float] = [],
        log_scale: Union[bool, Tuple[bool, bool]] = False,
        legend: bool = False,
        legend_loc: Union[str, Tuple] = 'upper left',
        n_cols: int = 1,
        markers: List[str] = [],
        markersizes: List[float] = [],
        markeredgewidths: List[float] = [],
        alphas: List[float] = [],
        colors: List[Optional[str]] = [],
        trend_methods: Union[str, List[str]] = 'none',
        trend_windows: List[int] = [],
        trend_spans: List[float] = [],
        trend_fracs: List[float] = [],
        trend_colors: List[Optional[str]] = [],
        trend_linestyles: List[str] = [],
        trend_linewidths: List[float] = [],
        trend_alphas: List[float] = [],
        trend_labels: List[str] = [],
        save_path: Optional[str] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        style: Optional[str] = "ggplot",
        return_ax: bool = False
    ) -> Optional[matplotlib.axes.Axes]:
    """Plot multiple series as scatter plots with optional per‑series trend lines.

    ``series_list`` is a list of integer lists; the ``i``th element of each
    list is plotted at x = i. Trend methods can be 'none', 'moving', 'ema',
    or 'lowess' (case-insensitive). Trend parameters (window/span/frac) are
    matched per series. If a single value is provided for a list of optional
    parameters, that value is applied to all series.
    """
    if not isinstance(series_list, list) or len(series_list) == 0:
        raise ValueError("series_list must be a non-empty list of lists.")
    G = len(series_list)
    def _get(lst, i, default):
        return lst[i] if isinstance(lst, list) and len(lst) > i and lst[i] is not None else default
    def _as_list(val, n, default):
        if isinstance(val, list):
            return [val[i] if i < len(val) else default for i in range(n)]
        return [val] * n
    trend_methods_arr = _as_list(trend_methods, G, 'none')
    fig, ax, created = _ensure_figure_ax(ax, (pic_width, pic_height), style=style)
    # smoothing helpers
    def _moving_avg(y: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return y.copy()
        if k % 2 == 0:
            k += 1
        pad = k // 2
        y_pad = np.pad(y, (pad, pad), mode='reflect')
        kernel = np.ones(k, dtype=float) / k
        sm = np.convolve(y_pad, kernel, mode='valid')
        return sm
    def _ema(y: np.ndarray, span: float) -> np.ndarray:
        if span <= 1:
            return y.copy()
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(y, dtype=float)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = alpha * y[i] + (1.0 - alpha) * out[i-1]
        return out
    def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.3, iters: int = 1) -> np.ndarray:
        n = len(x)
        if n == 0:
            return y
        frac = min(max(frac, 1.0 / n), 1.0)
        r = max(int(np.ceil(frac * n)), 2)
        y_fit = np.zeros(n, dtype=float)
        delta = np.ones(n, dtype=float)
        for _ in range(iters):
            for i in range(n):
                idx = np.argsort(np.abs(x - x[i]))[:r]
                x_nei = x[idx]
                y_nei = y[idx]
                dmax = np.max(np.abs(x_nei - x[i]))
                if dmax == 0:
                    y_fit[i] = y[i]
                    continue
                u = np.abs((x_nei - x[i]) / dmax)
                w = (1 - u ** 3) ** 3
                w = w * delta[idx]
                X = np.vstack([np.ones_like(x_nei), x_nei]).T
                Wsqrt = np.sqrt(w)
                Xw = X * Wsqrt[:, None]
                yw = y_nei * Wsqrt
                beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                y_fit[i] = beta[0] + beta[1] * x[i]
        return y_fit
    # Plot each series and optional trend
    for g in range(G):
        y_raw = list(series_list[g] if series_list[g] is not None else [])
        if len(y_raw) == 0:
            continue
        xvals = np.arange(len(y_raw), dtype=float)
        y = np.asarray(y_raw, dtype=float)
        marker = _get(markers, g, 'o')
        msize = _get(markersizes, g, 6.0)
        mew = _get(markeredgewidths, g, 1.5)
        alpha_pt = _get(alphas, g, 1.0)
        color_pt = _get(colors, g, None)
        label = _get(labels, g, "")
        scatter_kwargs: Dict[str, Any] = {
            "s": float(msize) ** 2,
            "marker": marker,
            "linewidths": mew,
            "alpha": alpha_pt,
            "zorder": 2,
        }
        if color_pt is not None:
            scatter_kwargs["c"] = color_pt
        ax.scatter(xvals, y, label=(label if legend and label else None), **scatter_kwargs)
        # trend line
        method = trend_methods_arr[g].lower() if isinstance(trend_methods_arr[g], str) else 'none'
        y_tr: Optional[np.ndarray] = None
        if method != 'none':
            if method == 'moving':
                k = int(_get(trend_windows, g, 5) or 5)
                y_tr = _moving_avg(y, k)
            elif method == 'ema':
                span = float(_get(trend_spans, g, 10) or 10)
                y_tr = _ema(y, span)
            elif method == 'lowess':
                frac = float(_get(trend_fracs, g, 0.3) or 0.3)
                y_tr = _lowess(xvals, y, frac=frac, iters=1)
        if y_tr is not None:
            t_color = _get(trend_colors, g, (color_pt if color_pt is not None else None))
            t_ls = _get(trend_linestyles, g, '-')
            t_lw = _get(trend_linewidths, g, 2.0)
            t_alpha = _get(trend_alphas, g, 0.9)
            default_tlabel = (f"{label} trend" if label else "trend")
            t_label = _get(trend_labels, g, default_tlabel)
            ax.plot(xvals, y_tr,
                    linestyle=t_ls,
                    linewidth=t_lw,
                    alpha=t_alpha,
                    color=t_color,
                    label=(t_label if legend and t_label else None),
                    zorder=3)
    # Titles and axis labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # axis ranges
    if x_range and len(x_range) == 2:
        ax.set_xlim(x_range)
    if y_range and len(y_range) == 2:
        ax.set_ylim(y_range)
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)
    # log scale options
    if isinstance(log_scale, bool):
        if log_scale:
            ax.set_yscale('log')
    elif isinstance(log_scale, tuple) and len(log_scale) == 2:
        if log_scale[0]:
            ax.set_xscale('log')
        if log_scale[1]:
            ax.set_yscale('log')
    ax.grid(True, linestyle="--", alpha=0.5)
    _legend(ax, legend, legend_loc, n_cols)
    fig.tight_layout()
    if save_path:
        save_fig(fig, title=title, save_path=save_path, dpi=150)
    if show:
        plt.show()
    else:
        if created:
            plt.close(fig)
    return ax if return_ax else None


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Sequence, Union, List, Optional, Literal


def save_legend_only_from_styles(
    labels: Sequence[str],
    colors: Sequence[str],
    markers: Sequence[str],
    filename: str,
    *,
    ncol: int = 1,
    linestyle: Union[str, Sequence[str]] = "-",
    linewidth: Union[float, Sequence[float]] = 3.0,
    markersize: Union[float, Sequence[float]] = 9.0,
    markeredgewidth: Union[float, Sequence[float]] = 2.0,
    # 外观
    frameon: bool = True,
    fancybox: Optional[bool] = None,
    columnspacing: float = 1.2,
    handletextpad: float = 0.6,
    labelspacing: float = 0.6,
    borderpad: float = 0.4,
    pad_inches: float = 0.02,
    dpi: int = 300,
    font_size: Optional[float] = None,
    # ✅ 新增：风格控制
    style: Literal["default", "ggplot"] = "default",
    # 可选：手动覆写边框与填充（若提供将优先于 style）
    edgecolor: Optional[str] = None,
    facecolor: Optional[str] = None,
    framealpha: Optional[float] = None,
    linewidth_frame: Optional[float] = None,
) -> str:
    """
    生成“只有 legend”的图，自动测量尺寸，无需裁剪。
    - 支持 style="ggplot"：浅灰填充、浅灰边框、实心不透明
    - 可用 edgecolor/facecolor/framealpha/linewidth_frame 手动覆写
    """
    if not (len(labels) == len(colors) == len(markers)):
        raise ValueError("labels, colors, markers 的长度必须一致")

    # 统一为列表（若传标量）
    def _as_list(x, n, cast=float):
        if isinstance(x, (list, tuple)):
            if len(x) != n:
                raise ValueError("参数长度不匹配")
            return [cast(v) for v in x]
        return [cast(x)] * n

    n = len(labels)
    markersize_list = _as_list(markersize, n, float)
    linewidth_list   = _as_list(linewidth, n, float)
    medgewidth_list  = _as_list(markeredgewidth, n, float)
    linestyle_list   = _as_list(linestyle, n, str)

    # 1) 构造 legend handles
    handles: List[Line2D] = []
    for lab, c, m, ms, lw, mew, ls in zip(
        labels, colors, markers,
        markersize_list, linewidth_list, medgewidth_list, linestyle_list
    ):
        handles.append(Line2D(
            [0],[0], label=lab,
            color=c, linestyle=ls, linewidth=lw,
            marker=m, markersize=ms, markeredgewidth=mew
        ))

    # 2) 画布与 legend
    fig, ax = plt.subplots(figsize=(0.01, 0.01), dpi=dpi)
    ax.set_axis_off()
    kw = dict(
        handles=handles, labels=labels, loc="center", ncol=ncol,
        frameon=frameon, borderpad=borderpad, columnspacing=columnspacing,
        handletextpad=handletextpad, labelspacing=labelspacing
    )
    if fancybox is not None:
        kw["fancybox"] = fancybox
    leg = ax.legend(**kw)

    # 3) 应用风格（ggplot or default），可被显式参数覆写
    frm = leg.get_frame()
    if style == "ggplot":
        # 近似 matplotlib ggplot 风格：legend 继承 axes.facecolor，边框浅灰
        gg_axes_face = plt.rcParams.get("axes.facecolor", "0.9")
        gg_edge      = plt.rcParams.get("axes.edgecolor", "0.8")
        frm.set_facecolor(gg_axes_face)
        frm.set_edgecolor(gg_edge)
        frm.set_alpha(1.0)
        frm.set_linewidth(1.0)

    # 显式覆写优先
    if facecolor is not None:
        frm.set_facecolor(facecolor)
    if edgecolor is not None:
        frm.set_edgecolor(edgecolor)
    if framealpha is not None:
        frm.set_alpha(framealpha)
    if linewidth_frame is not None:
        frm.set_linewidth(linewidth_frame)

    if font_size is not None:
        for t in leg.get_texts(): t.set_fontsize(font_size)

    # 4) 自适应尺寸与保存
    fig.canvas.draw()
    bbox = leg.get_window_extent(renderer=fig.canvas.get_renderer())
    fig.set_size_inches(bbox.width/dpi, bbox.height/dpi)
    fig.savefig(filename, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return filename

from typing import Sequence, List, Optional, Union, Literal
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def save_bar_legend_only_from_styles(
    serie_labels: Sequence[str],
    filename: str,
    *,
    # —— 与 draw_grouped_bar 的风格输入保持一致 —— #
    colors: Sequence[str] = (),        # 外边框颜色（edgecolor），若长度 != n 则不强制设置
    hatches: Sequence[str] = (),       # 柱体填充纹理（hatch），若长度 != n 则不设置
    inner_colors: Sequence[str] = (),  # 柱体面色（facecolor），若长度 != n 则回退到颜色循环
    # —— 线宽（对应你画图里 legend=True 时的 linewidth=2） —— #
    edge_linewidth: Union[float, Sequence[float]] = 2.0,
    # —— Legend 布局 —— #
    ncol: int = 1,
    frameon: bool = True,
    fancybox: Optional[bool] = None,
    columnspacing: float = 1.2,
    handletextpad: float = 0.6,
    labelspacing: float = 0.6,
    borderpad: float = 0.4,
    pad_inches: float = 0.02,
    dpi: int = 300,
    font_size: Optional[float] = None,
    # —— 风格：与折线 legend 的 ggplot 方案一致，可被下方覆写参数覆盖 —— #
    style: Literal["default", "ggplot"] = "ggplot",
    # 手动覆写 legend 框（优先级最高）
    edgecolor: Optional[str] = None,       # legend 框边框色
    facecolor: Optional[str] = None,       # legend 框底色
    framealpha: Optional[float] = None,    # legend 框透明度
    linewidth_frame: Optional[float] = None, # legend 框线宽
) -> str:
    """
    Create a *legend-only* figure for grouped bars and save it to `filename`.

    Parameters mirror the aesthetics of `draw_grouped_bar`:
    - `colors` -> bar edgecolor
    - `inner_colors` -> bar facecolor
    - `hatches` -> bar hatch
    If a list length doesn't match `len(serie_labels)`, that attribute is left to defaults
    (facecolor will fall back to the current color cycle).
    """
    n = len(serie_labels)
    if n == 0:
        raise ValueError("`serie_labels` 不能为空")

    def _as_list(x, n, cast_func=lambda v: v):
        if isinstance(x, (list, tuple)):
            if len(x) != n:
                raise ValueError("参数长度不匹配")
            return [cast_func(v) for v in x]
        return [cast_func(x)] * n

    # 处理 edge_linewidth，其他三者按“长度匹配才生效”的策略与 draw_grouped_bar 一致
    lw_list = _as_list(edge_linewidth, n, float)

    # facecolor 回退到颜色循环（保证各系列有可见区分）
    def _cycle_colors(k: int) -> List[str]:
        prop = plt.rcParams.get("axes.prop_cycle")
        if prop is not None and "color" in prop._mapping:
            cyc = list(prop.by_key().get("color", []))
            if not cyc:
                cyc = ["#1f77b4"]
        else:
            cyc = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                   "#bcbd22", "#17becf"]
        out = []
        for i in range(k):
            out.append(cyc[i % len(cyc)])
        return out

    use_ec = list(colors) if len(colors) == n else [None] * n
    use_hatch = list(hatches) if len(hatches) == n else [None] * n
    if len(inner_colors) == n:
        use_fc = list(inner_colors)
    else:
        # 与 Matplotlib 对多系列默认上色的行为近似：为不同系列分配循环色
        use_fc = _cycle_colors(n)

    # 构造 legend 代理句柄（每个系列一个 Patch）
    handles: List[Patch] = []
    for lab, fc, ec, ht, lw in zip(serie_labels, use_fc, use_ec, use_hatch, lw_list):
        handles.append(Patch(
            label=lab,
            facecolor=fc,
            edgecolor=ec,
            hatch=ht,
            linewidth=lw
        ))

    # 生成最小画布并放置 legend
    fig, ax = plt.subplots(figsize=(0.01, 0.01), dpi=dpi)
    ax.set_axis_off()
    kw = dict(
        handles=handles,
        labels=serie_labels,
        loc="center",
        ncol=ncol,
        frameon=frameon,
        borderpad=borderpad,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        labelspacing=labelspacing,
    )
    if fancybox is not None:
        kw["fancybox"] = fancybox
    leg = ax.legend(**kw)

    # 应用 ggplot 风格（可被显式覆写参数覆盖）
    frm = leg.get_frame()
    if style == "ggplot":
        gg_axes_face = plt.rcParams.get("axes.facecolor", "0.9")
        gg_edge = plt.rcParams.get("axes.edgecolor", "0.8")
        frm.set_facecolor(gg_axes_face)
        frm.set_edgecolor(gg_edge)
        frm.set_alpha(1.0)
        frm.set_linewidth(1.0)

    if facecolor is not None:
        frm.set_facecolor(facecolor)
    if edgecolor is not None:
        frm.set_edgecolor(edgecolor)
    if framealpha is not None:
        frm.set_alpha(framealpha)
    if linewidth_frame is not None:
        frm.set_linewidth(linewidth_frame)

    if font_size is not None:
        for t in leg.get_texts():
            t.set_fontsize(font_size)

    # 自适应 legend 尺寸并保存
    fig.canvas.draw()
    bbox = leg.get_window_extent(renderer=fig.canvas.get_renderer())
    fig.set_size_inches(bbox.width / dpi, bbox.height / dpi)
    fig.savefig(filename, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return filename
