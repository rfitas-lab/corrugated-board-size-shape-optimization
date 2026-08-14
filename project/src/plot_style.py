"""Shared publication style for every Python-generated manuscript figure.

Python graphics are deliberately monochrome.  Red and gold are reserved for
explicitly selected designs or observations.  Text uses the installed Latin
Modern family and Computer-Modern mathematical glyphs, matching LaTeX without
depending on Matplotlib's external TeX wrapper.
"""

from __future__ import annotations

import matplotlib as mpl

BLACK = "#000000"
GRAY_DARK = "#4D4D4D"
GRAY = "#808080"
GRAY_LIGHT = "#D9D9D9"
GRAY_PALE = "#F2F2F2"
HIGHLIGHT_RED = "#B2182B"
HIGHLIGHT_GOLD = "#D39C00"


def apply_latex_style(font_size: float = 8.0) -> None:
    """Apply a compact LaTeX/Computer-Modern publication style."""

    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": max(font_size - 0.7, 5.5),
            "xtick.labelsize": max(font_size - 0.8, 5.5),
            "ytick.labelsize": max(font_size - 0.8, 5.5),
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.unicode_minus": False,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
