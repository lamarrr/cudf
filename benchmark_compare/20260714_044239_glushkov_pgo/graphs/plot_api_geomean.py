#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the cuDF regex interpreter-versus-JIT API geometric means."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cudf-regex")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogLocator, NullFormatter

OUTPUT = Path(__file__).resolve().parent
SUMMARY = OUTPUT / "summary.csv"

GREEN = "#76B900"
GREEN_BRIGHT = "#B8F34A"
CHARCOAL = "#0B0B0B"
PANEL = "#181818"
PANEL_ALT = "#222222"
WHITE = "#F5F5F5"
GRAY = "#A7A7A7"
GRID = "#444444"


def load_summary() -> list[dict[str, str]]:
    with SUMMARY.open() as source:
        return list(csv.DictReader(source))


def speedup_label(value: float) -> str:
    return f"{value:.2f}×"


def render() -> None:
    rows = load_summary()
    labels = [row["label"] for row in rows]
    interpreter = [float(row["interpreter_ms"]) for row in rows]
    jit = [float(row["jit_ms"]) for row in rows]
    states = [int(row["states"]) for row in rows]
    positions = list(range(len(rows)))
    width = 0.34
    y_min = min(interpreter + jit) * 0.45
    y_max = max(interpreter + jit) * 2.6

    figure, axis = plt.subplots(figsize=(13.5, 6.4))
    figure.patch.set_facecolor(CHARCOAL)
    figure.add_artist(
        Rectangle(
            (0.025, 0.963),
            0.045,
            0.009,
            transform=figure.transFigure,
            color=GREEN,
            linewidth=0,
        )
    )
    figure.text(
        0.025,
        0.982,
        "RAPIDS cuDF",
        color=GREEN_BRIGHT,
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.16,
        0.982,
        "REGEX API GEOMETRIC MEAN",
        color=WHITE,
        fontsize=19,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.16,
        0.951,
        "One geometric mean across every parameterized state within each cuDF regex API",
        color=GRAY,
        fontsize=9,
        va="top",
    )
    figure.text(
        0.975,
        0.955,
        "LOWER IS BETTER ↓",
        color=GREEN_BRIGHT,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": PANEL_ALT,
            "edgecolor": GREEN,
        },
    )
    figure.text(
        0.975,
        0.010,
        "RTX A6000  •  NVBench GPU mean  •  compilation excluded",
        color="#777777",
        fontsize=7,
        ha="right",
    )

    axis.set_facecolor(PANEL)
    axis.bar(
        [position - width / 2 for position in positions],
        [value - y_min for value in interpreter],
        width,
        bottom=y_min,
        color=GRAY,
        edgecolor="#D0D0D0",
        linewidth=0.7,
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [value - y_min for value in jit],
        width,
        bottom=y_min,
        color=GREEN,
        edgecolor=GREEN_BRIGHT,
        linewidth=0.7,
    )
    for position, interpreter_time, jit_time, state_count in zip(
        positions, interpreter, jit, states, strict=True
    ):
        axis.text(
            position,
            max(interpreter_time, jit_time) * 1.13,
            speedup_label(interpreter_time / jit_time),
            color=GREEN_BRIGHT,
            fontsize=12,
            fontweight="bold",
            ha="center",
        )
        axis.text(
            position,
            y_min * 1.25,
            f"n={state_count}",
            color="#777777",
            fontsize=8,
            ha="center",
        )

    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(
        axis="y",
        which="major",
        color=GRID,
        linewidth=0.7,
        linestyle="--",
        alpha=0.7,
    )
    axis.set_axisbelow(True)
    axis.set_xticks(positions, labels)
    axis.set_ylim(y_min, y_max)
    axis.set_ylabel(
        "Geometric-mean GPU execution time (ms, log scale)",
        color=WHITE,
        fontsize=10,
    )
    axis.tick_params(colors=GRAY, labelsize=8, length=0)
    axis.tick_params(axis="x", labelsize=10, colors=WHITE)
    for spine in axis.spines.values():
        spine.set_color("#303030")
    axis.legend(
        handles=[
            Patch(facecolor=GRAY, edgecolor="#D0D0D0", label="Interpreter"),
            Patch(facecolor=GREEN, edgecolor=GREEN_BRIGHT, label="NVVM JIT"),
        ],
        loc="upper right",
        frameon=False,
        labelcolor=WHITE,
        fontsize=10,
        ncol=2,
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.12)
    for extension in ("png", "svg"):
        output_path = OUTPUT / f"api_geomean.{extension}"
        figure.savefig(
            output_path,
            dpi=220,
            facecolor=figure.get_facecolor(),
        )
        if extension == "svg":
            output_path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in output_path.read_text().splitlines()
                )
                + "\n"
            )
    plt.close(figure)


if __name__ == "__main__":
    render()
