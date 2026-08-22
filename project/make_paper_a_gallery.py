#!/usr/bin/env python3
"""Build the geometric Pareto atlas and selected-design galleries."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plot_style import (  # noqa: E402
    BLACK,
    GRAY,
    GRAY_LIGHT,
    HIGHLIGHT_GOLD,
    HIGHLIGHT_RED,
    apply_latex_style,
)

DATA_ROOT = ROOT / "data" / "paper_a" / "opt_cbv2_final_fronts"
FIGURE_ROOT = ROOT / "figures" / "paper_a"
TABLE_ROOT = ROOT / "manuscripts" / "paper_a" / "generated_tables"

SELECT_BLUE = "#1F5A94"
SELECT_RED = "#C8323A"
SELECT_GREEN = "#2A8C62"

ROW_DEFINITIONS = [
    ("4 control points", [19, 20, 21], "o", "-"),
    ("8 control points", [22, 23, 24], "s", "--"),
    ("15 control points", [25, 26, 27], "D", ":"),
    (r"7 variables, $R_{\min}>0$", [28, 29, 30], "^", "-."),
    (r"7 variables, $R_{\min}>0.9$ mm", [31, 32, 33], "v", (0, (5, 1, 1, 1))),
]

COLUMN_TITLES = [
    r"$E_{z,\mathrm{eff}}$ vs. $I/\lambda$",
    r"$I/\lambda$ vs. $A/\lambda$",
    r"$A/(\lambda E_{z,\mathrm{eff}})$ vs. $A/I$",
]

COLUMN_LABELS = [
    (r"$E_{z,\mathrm{eff}}$", r"$I/\lambda$"),
    (r"$I/\lambda$", r"$A/\lambda$"),
    (r"$A/(\lambda E_{z,\mathrm{eff}})$", r"$A/I$"),
]


def parse_vector(value: str) -> np.ndarray:
    cleaned = str(value).replace("np.float64(", "").replace("numpy.float64(", "")
    cleaned = re.sub(r"\)(?=[,\]\s])", "", cleaned)
    try:
        return np.asarray(ast.literal_eval(cleaned), dtype=float)
    except Exception:
        numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(value))
        return np.asarray([float(item) for item in numbers], dtype=float)


def load_case(case_number: int) -> pd.DataFrame:
    path = DATA_ROOT / f"C{case_number:03d}_final.csv"
    if not path.exists():
        raise FileNotFoundError(f"No numerical front was found for C{case_number:03d}")
    data = pd.read_csv(path)
    data = data.rename(columns={"xline_all_values": "x", "yline_all_values": "y"})
    data = data[np.isfinite(data.x) & np.isfinite(data.y)].copy()
    data = data.drop_duplicates(subset=["x", "y"]).sort_values("x")
    return data.reset_index(drop=True)


def triplet_profile(vector: np.ndarray, samples: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the [y1,x2,w2,y2,...] rational curve."""

    if len(vector) < 4 or (len(vector) - 1) % 3:
        raise ValueError("Triplet parameterization requires 3n+1 variables")
    x_steps = np.asarray(vector[1::3], dtype=float)
    y_values = np.asarray(vector[0::3], dtype=float)
    raw_weights = np.asarray(vector[2::3], dtype=float)
    normalizer = x_steps.sum() + 1.0 / max(1, len(x_steps))
    control_x = np.concatenate(([0.0], np.cumsum(x_steps) / normalizer, [1.0]))
    control_y = np.concatenate((y_values, [y_values[0]]))
    weights = np.concatenate(
        ([1.0], 0.0103 * np.exp(9.17 * raw_weights) + 0.9897, [1.0])
    )
    degree = 2
    count = len(control_x)
    inner_count = count - degree - 1
    knots = np.concatenate(
        [np.zeros(degree + 1), np.linspace(0, 1, inner_count + 2)[1:-1], np.ones(degree + 1)]
    )
    parameter = np.linspace(0.0, 1.0, samples)
    basis = bspline_basis(parameter, knots, count, degree)
    weighted = basis * weights[None, :]
    denominator = np.maximum(weighted.sum(axis=1), 1e-14)
    x = weighted @ control_x / denominator
    y = weighted @ control_y / denominator
    x = 5.65 * (x - x[0]) / max(x[-1] - x[0], 1e-14)
    y = 2.65 * (y - y.min()) / max(np.ptp(y), 1e-14)
    return x, y


def size_profile(vector: np.ndarray, samples: int = 1200) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return one sinusoidal size-design period and its sheet thicknesses."""

    if len(vector) == 5:
        t_bottom, t_flute, wavelength, height, t_top = map(float, vector)
    elif len(vector) == 4:
        t_bottom, t_flute, wavelength, height = map(float, vector)
        t_top = t_bottom
    elif len(vector) == 2:
        wavelength, height = map(float, vector)
        t_bottom = t_flute = t_top = 0.2
    else:
        raise ValueError("Size parameterization requires 2, 4 or 5 variables")
    x = np.linspace(0.0, max(wavelength, 1e-6), samples)
    centre_min = t_bottom + 0.5 * t_flute
    centre_max = t_bottom + height - 0.5 * t_flute
    middle = 0.5 * (centre_min + centre_max)
    amplitude = 0.5 * (centre_max - centre_min)
    y = middle - amplitude * np.cos(2.0 * np.pi * x / max(wavelength, 1e-6))
    return x, y, t_bottom, t_flute, t_top


def draw_design(ax: plt.Axes, vector: np.ndarray, repetitions: int = 3) -> None:
    """Draw a selected design in physical millimetres."""

    if len(vector) in (2, 4, 5):
        x_one, y_one, t_bottom, t_flute, t_top = size_profile(vector)
        wavelength = float(x_one[-1])
        height_total = float(y_one.max() + 0.5 * t_flute + t_top)
    elif len(vector) in (13, 25, 46):
        x_one, y_zero = triplet_profile(vector)
        t_bottom = t_flute = t_top = 0.2
        y_one = y_zero + t_bottom + 0.5 * t_flute
        wavelength = 5.65
        height_total = 2.65 + t_bottom + t_flute + t_top
    elif len(vector) in (7, 10):
        x_one, y_zero = seven_variable_profile(vector[:7])
        if len(vector) == 10:
            wavelength, amplitude, thickness = map(float, vector[7:10])
            x_one *= wavelength / 5.65
            y_zero *= amplitude / 2.65
            t_bottom = t_flute = t_top = thickness
        else:
            wavelength, amplitude = 5.65, 2.65
            t_bottom = t_flute = t_top = 0.2
        y_one = y_zero + t_bottom + 0.5 * t_flute
        height_total = amplitude + t_bottom + t_flute + t_top
    else:
        raise ValueError(f"Unsupported design-vector length {len(vector)}")

    for repetition in range(repetitions):
        ax.plot(x_one + repetition * wavelength, y_one, color=BLACK, linewidth=0.9)
    width = repetitions * wavelength
    ax.plot([0, width], [0.5 * t_bottom, 0.5 * t_bottom], color=BLACK, linewidth=1.5)
    ax.plot(
        [0, width],
        [height_total - 0.5 * t_top, height_total - 0.5 * t_top],
        color=BLACK,
        linewidth=1.5,
    )
    ax.set_xlim(-0.01 * width, 1.01 * width)
    ax.set_ylim(-0.05 * height_total, 1.05 * height_total)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(0.20)
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.45)


def compact_vector(vector: np.ndarray) -> str:
    """Page-safe numerical vector with explicit omission marks for high dimensions."""

    if len(vector) <= 10:
        values = vector
        return "(" + ", ".join(f"{value:.3g}" for value in values) + ")"
    values = np.concatenate((vector[:5], vector[-2:]))
    return "(" + ", ".join(f"{value:.3g}" for value in values[:5]) + ", \\ldots, " + ", ".join(f"{value:.3g}" for value in values[-2:]) + ")"


def write_design_table(stem: str, cases: list[int]) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_number in cases:
        data = load_case(case_number)
        for solution, index in zip(("S1", "S2", "S3"), selected_indices(data)):
            vector = parse_vector(data.X_all_values.iloc[index])
            rows.append(
                rf"C{case_number:03d} & {solution} & {data.x.iloc[index]:.4g} & {data.y.iloc[index]:.4g} & ${compact_vector(vector)}$ \\"
            )
    content = "\n".join(
        [
            r"\begin{table*}[p]",
            r"\centering\scriptsize",
            rf"\caption{{Numerical selected-design record associated with Fig.~\ref{{fig:{stem}}}. S1 is the first-objective extreme, S2 the log-normalized compromise and S3 the second-objective extreme. For high-dimensional control-point vectors, the printed endpoint components provide a compact shape summary.}}",
            rf"\label{{tab:{stem}}}",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\begin{tabularx}{\textwidth}{llrr>{\raggedright\arraybackslash}X}",
            r"\toprule Case & Design & $f_1$ & $f_2$ & Design vector (mm where dimensional) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    (TABLE_ROOT / f"{stem}.tex").write_text(content, encoding="utf-8")


def scientific_ticks(ax: plt.Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        axis.set_major_formatter(formatter)
        axis.set_major_locator(MaxNLocator(3))
    ax.grid(True, color=GRAY_LIGHT, linewidth=0.35, linestyle=":", zorder=0)
    ax.tick_params(pad=1.5)


def pareto_atlas() -> None:
    apply_latex_style(8.2)
    fig, axes = plt.subplots(5, 3, figsize=(7.15, 8.55), constrained_layout=False)
    for row, (row_name, case_numbers, marker, linestyle) in enumerate(ROW_DEFINITIONS):
        for column, case_number in enumerate(case_numbers):
            ax = axes[row, column]
            data = load_case(case_number)
            ax.plot(
                data.x,
                data.y,
                color=BLACK,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor="white",
                markeredgecolor=BLACK,
                markeredgewidth=0.45,
                markersize=2.7,
                linewidth=0.75,
                zorder=2,
            )
            scientific_ticks(ax)
            ax.text(
                0.97,
                0.94,
                rf"C{case_number:03d}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.4,
            )
            if row == 0:
                ax.set_title(COLUMN_TITLES[column], pad=4.0, fontsize=8.4)
            if row == 4:
                ax.set_xlabel(COLUMN_LABELS[column][0], labelpad=2.0)
            if column == 0:
                ax.set_ylabel(COLUMN_LABELS[column][1], labelpad=2.0)
                ax.annotate(
                    row_name,
                    xy=(-0.38, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8.1,
                )
    legend_items = [
        Line2D([0], [0], color=BLACK, linestyle=style, marker=marker,
               markerfacecolor="white", markeredgecolor=BLACK, markersize=3.2,
               linewidth=0.8, label=name)
        for name, _, marker, style in ROW_DEFINITIONS
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.003),
        handlelength=2.6,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=0.965, bottom=0.075, hspace=0.42, wspace=0.33)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / "pareto_atlas_5x3.pdf")
    fig.savefig(FIGURE_ROOT / "pareto_atlas_5x3.png", dpi=420)
    plt.close(fig)


def bspline_basis(parameters: np.ndarray, knots: np.ndarray, count: int, degree: int) -> np.ndarray:
    basis = np.zeros((len(parameters), count + degree))
    for index in range(count + degree):
        basis[:, index] = ((parameters >= knots[index]) & (parameters < knots[index + 1])).astype(float)
    basis[-1, :] = 0.0
    basis[-1, count - 1] = 1.0
    for order in range(1, degree + 1):
        updated = np.zeros_like(basis)
        for index in range(count + degree - order):
            left_denominator = knots[index + order] - knots[index]
            right_denominator = knots[index + order + 1] - knots[index + 1]
            if left_denominator > 0:
                updated[:, index] += (parameters - knots[index]) / left_denominator * basis[:, index]
            if right_denominator > 0:
                updated[:, index] += (knots[index + order + 1] - parameters) / right_denominator * basis[:, index + 1]
        basis = updated
    return basis[:, :count]


def seven_variable_profile(vector: np.ndarray, samples: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    if len(vector) != 7:
        raise ValueError("The constrained profile requires seven variables")
    fractions = np.maximum(vector[:5], 1e-12)
    fractions = fractions / fractions.sum()
    d1, d2, d3, d4, _ = fractions
    control_x = np.asarray(
        [0.0, d1, d1 + d2, d1 + d2 + d3, d1 + d2 + d3 + d4,
         1.0 + d1, 1.0 + d1 + d2, 1.0 + d1 + d2 + d3,
         1.0 + d1 + d2 + d3 + d4, 2.0]
    )
    control_y = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    weight_1 = 0.0103 * np.exp(9.17 * vector[5]) + 0.1
    weight_2 = 0.0103 * np.exp(9.17 * vector[6]) + 0.1
    weights = np.asarray([1, weight_1, weight_1, weight_2, weight_2,
                          weight_1, weight_1, weight_2, weight_2, 1], dtype=float)
    degree = 2
    count = len(control_x)
    inner_count = count - degree - 1
    knots = np.concatenate(
        [np.zeros(degree + 1), np.linspace(0, 1, inner_count + 2)[1:-1], np.ones(degree + 1)]
    )
    parameter = np.linspace(0.0, 1.0, samples)
    basis = bspline_basis(parameter, knots, count, degree)
    weighted = basis * weights[None, :]
    denominator = np.maximum(weighted.sum(axis=1), 1e-14)
    curve_x = weighted @ control_x / denominator
    curve_y = weighted @ control_y / denominator
    start_fraction = (d1 + d2 + d3 / 2.0) / 2.0
    end_fraction = (1.0 + d1 + d2 + d3 / 2.0) / 2.0
    mask = (parameter >= start_fraction) & (parameter <= end_fraction)
    curve_x = curve_x[mask]
    curve_y = curve_y[mask]
    curve_x = 5.65 * (curve_x - curve_x[0]) / max(curve_x[-1] - curve_x[0], 1e-14)
    curve_y = 2.65 * (curve_y - curve_y.min()) / max(np.ptp(curve_y), 1e-14)
    return curve_x, curve_y


def selected_indices(data: pd.DataFrame) -> list[int]:
    x = np.log(np.maximum(data.x.to_numpy(), 1e-15))
    y = np.log(np.maximum(data.y.to_numpy(), 1e-15))
    xn = (x - x.min()) / max(np.ptp(x), 1e-15)
    yn = (y - y.min()) / max(np.ptp(y), 1e-15)
    left = int(np.argmin(xn))
    right = int(np.argmin(yn))
    knee = int(np.argmin(np.sqrt(xn**2 + yn**2)))
    return [left, knee, right]


def constrained_design_gallery() -> None:
    apply_latex_style(7.2)
    fig = plt.figure(figsize=(7.15, 4.75))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.65, 0.50, 0.50, 0.50], hspace=0.28, wspace=0.25)
    labels = [r"minimum first objective", r"log-space knee", r"minimum second objective"]
    # The three selections must remain distinguishable in print and for
    # readers with impaired colour vision: one neutral extreme, a red knee,
    # and one gold extreme.  Shape is redundant with colour.
    point_markers = ["s", "D", "^"]
    point_colours = [SELECT_BLUE, SELECT_RED, SELECT_GREEN]
    for column, case_number in enumerate([31, 32, 33]):
        data = load_case(case_number)
        indices = selected_indices(data)
        ax_front = fig.add_subplot(grid[0, column])
        ax_front.plot(data.x, data.y, color=BLACK, linestyle="-", linewidth=0.8)
        ax_front.scatter(data.x, data.y, facecolor="white", edgecolor=BLACK,
                         marker="v", linewidth=0.45, s=9, zorder=2)
        for label_index, (index, marker, colour) in enumerate(zip(indices, point_markers, point_colours)):
            ax_front.scatter(data.x.iloc[index], data.y.iloc[index], marker=marker,
                             facecolor=colour, edgecolor=BLACK, linewidth=0.6, s=34,
                             zorder=4, label=labels[label_index])
        scientific_ticks(ax_front)
        ax_front.set_title(rf"C{case_number:03d}: {COLUMN_TITLES[column]}", fontsize=7.4)
        ax_front.set_xlabel(COLUMN_LABELS[column][0])
        ax_front.set_ylabel(COLUMN_LABELS[column][1])
        for row, (index, marker, colour) in enumerate(zip(indices, point_markers, point_colours), start=1):
            ax_shape = fig.add_subplot(grid[row, column])
            vector = parse_vector(data.X_all_values.iloc[index])
            draw_design(ax_shape, vector)
            if column == 0:
                ax_shape.set_ylabel(labels[row - 1], fontsize=6.5)
    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.96, bottom=0.11)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / "constrained_selected_designs.pdf")
    fig.savefig(FIGURE_ROOT / "constrained_selected_designs.png", dpi=420)
    plt.close(fig)


CASE_PLATES = [
    ("Full five-variable size optimization", [1, 2, 3, 4]),
    ("Envelope-only size optimization", [5, 6, 7, 8]),
    ("Four-control-point shape optimization", [9, 10, 11]),
    ("Eight-control-point shape optimization", [12, 13, 14]),
    ("Fifteen-control-point shape optimization", [15, 16, 17]),
    ("Legacy compact rational-shape case", [18]),
    ("Compact rational-shape campaign", [28, 29, 30]),
    ("Production four-control-point campaign", [19, 20, 21]),
    ("Production eight-control-point campaign", [22, 23, 24]),
    ("Production fifteen-control-point campaign", [25, 26, 27]),
    (r"Radius-constrained compact rational campaign", [31, 32, 33]),
]


def axis_labels_for_case(case_number: int) -> tuple[str, str]:
    if case_number in {1, 3, 5, 7, 9, 12, 15, 18, 21, 24, 25, 26, 27, 28, 29, 30}:
        return r"$A/(\lambda E_{z,\mathrm{eff}})$", r"$A/I$"
    if case_number in {2, 4, 6, 8, 10, 13, 16, 19, 22, 31, 33}:
        return r"$E_{z,\mathrm{eff}}$", r"$I/\lambda$"
    if case_number in {11, 14, 17, 20, 23, 32}:
        return r"$I/\lambda$", r"$A/\lambda$"
    if case_number == 34:
        return r"$A/(5\lambda)$", r"$10^3/E_{z,\mathrm{eff}}$"
    if case_number == 35:
        return r"$A/(5\lambda)$", r"$1/(I/\lambda)$"
    if case_number == 36:
        return r"$100A/(\lambda E_{z,\mathrm{eff}})$", r"$A/I$"
    return r"$f_1$", r"$f_2$"


def pareto_family_plates() -> None:
    """Pareto plus S1/S2/S3 geometries for every case family."""

    apply_latex_style(7.2)
    selection_markers = ["o", "D", "^"]
    selection_colours = [SELECT_BLUE, SELECT_RED, SELECT_GREEN]
    selection_labels = [
        "S1: first-objective extreme",
        "S2: log-normalized knee",
        "S3: second-objective extreme",
    ]
    for family_index, (family, cases) in enumerate(CASE_PLATES, start=1):
        columns = len(cases)
        fig = plt.figure(figsize=(7.15, 5.25 if columns == 4 else 5.0))
        grid = fig.add_gridspec(4, columns, height_ratios=[1.70, 0.50, 0.50, 0.50], hspace=0.30, wspace=0.30)
        first_front = None
        for column, case_number in enumerate(cases):
            data = load_case(case_number)
            ax = fig.add_subplot(grid[0, column])
            if first_front is None:
                first_front = ax
            ax.plot(data.x, data.y, color=BLACK, linestyle="-", linewidth=0.8)
            ax.scatter(data.x, data.y, marker="o", facecolor="white", edgecolor=BLACK, linewidth=0.4, s=8)
            indices = selected_indices(data)
            for label, index, point_marker, point_colour in zip(
                selection_labels, indices, selection_markers, selection_colours
            ):
                ax.scatter(
                    data.x.iloc[index], data.y.iloc[index], marker=point_marker,
                    facecolor=point_colour, edgecolor=BLACK, linewidth=0.5,
                    s=30, zorder=4, label=label,
                )
            scientific_ticks(ax)
            xlabel, ylabel = axis_labels_for_case(case_number)
            ax.set_title(rf"C{case_number:03d}", fontsize=7.4)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            for row, (solution, index, colour) in enumerate(zip(("S1", "S2", "S3"), indices, selection_colours), start=1):
                shape_ax = fig.add_subplot(grid[row, column])
                vector = parse_vector(data.X_all_values.iloc[index])
                draw_design(shape_ax, vector)
                shape_ax.text(
                    0.015, 0.83, solution, transform=shape_ax.transAxes,
                    color=colour, fontweight="bold", ha="left", va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.88},
                )
        handles, labels = first_front.get_legend_handles_labels() if first_front is not None else ([], [])
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(family, y=0.992)
        fig.subplots_adjust(left=0.09, right=0.995, top=0.93, bottom=0.09)
        stem = f"case_plate_{family_index:02d}"
        fig.savefig(FIGURE_ROOT / f"{stem}.pdf")
        fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=420)
        plt.close(fig)
        write_design_table(stem, cases)


def coupled_plate() -> None:
    cases = [34, 35, 36]
    if not all((DATA_ROOT / f"C{case:03d}_final.csv").exists() for case in cases):
        return
    CASE_PLATES.append(("New coupled size--shape optimization", cases))
    # Generate only the appended plate without repeating the first ten.
    family_index = len(CASE_PLATES)
    family, cases = CASE_PLATES[-1]
    apply_latex_style(7.2)
    fig = plt.figure(figsize=(7.15, 5.0))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.70, 0.50, 0.50, 0.50], hspace=0.30, wspace=0.30)
    colours = [SELECT_BLUE, SELECT_RED, SELECT_GREEN]
    markers = ["o", "D", "^"]
    labels = ["S1: first-objective extreme", "S2: log-normalized knee", "S3: second-objective extreme"]
    front_axes = []
    for column, case_number in enumerate(cases):
        data = load_case(case_number)
        indices = selected_indices(data)
        ax = fig.add_subplot(grid[0, column]); front_axes.append(ax)
        ax.plot(data.x, data.y, color=BLACK, linewidth=0.8)
        ax.scatter(data.x, data.y, facecolor="white", edgecolor=BLACK, s=8, linewidth=0.4)
        for label, index, marker, colour in zip(labels, indices, markers, colours):
            ax.scatter(data.x.iloc[index], data.y.iloc[index], marker=marker, facecolor=colour, edgecolor=BLACK, s=30, linewidth=0.5, label=label, zorder=4)
        scientific_ticks(ax)
        xlabel, ylabel = axis_labels_for_case(case_number)
        ax.set_title(rf"C{case_number:03d}")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        for row, (solution, index, colour) in enumerate(zip(("S1", "S2", "S3"), indices, colours), start=1):
            shape_ax = fig.add_subplot(grid[row, column])
            draw_design(shape_ax, parse_vector(data.X_all_values.iloc[index]))
            shape_ax.text(0.015, 0.83, solution, transform=shape_ax.transAxes, color=colour, fontweight="bold", ha="left", va="top", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.88})
    handles, legend_labels = front_axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(family, y=0.992)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.93, bottom=0.09)
    stem = f"case_plate_{family_index:02d}"
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf"); fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=420)
    plt.close(fig)
    write_design_table(stem, cases)


if __name__ == "__main__":
    pareto_atlas()
    constrained_design_gallery()
    pareto_family_plates()
    coupled_plate()
