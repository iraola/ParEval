"""Shared plotting style for the analysis figures.

Single source of truth for figure aesthetics: each model gets a fixed
colour, marker, and short label (hues grouped by model family), so it looks
the same in every figure regardless of which subset is plotted or how it is
ordered. Also provides the common axis styling, save helpers (PDF + PNG),
model-subset presets, and path/IO helpers used by the plot-*.py scripts.

To add a new model, append its exact `model`-column name to the right family
in FAMILIES; colours and markers are assigned deterministically from that
list. Unknown models fall back to grey with a warning.
"""
import colorsys
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Model registry ────────────────────────────────────────────────────────────

FAMILIES: dict[str, dict] = {
    "Qwen":     {"base": "#C0392B", "models": [
        "Qwen3-32B", "Qwen2.5-Coder-32B-Instruct", "Qwen2.5-32B-Instruct"]},
    "DeepSeek": {"base": "#2273D6", "models": [
        "DeepSeek-R1-Distill-Qwen-32B", "DeepSeek-R1-Distill-Llama-70B",
        "DeepSeek-Coder-V2-Lite-Base", "deepseek-coder-6.7b-instruct",
        "deepseek-coder-6.7b-base"]},
    "Llama":    {"base": "#1BB863", "models": [
        "Llama-3.3-70B-Instruct", "Llama-3.1-8B-Instruct"]},
    "CodeLlama":{"base": "#12A190", "models": [
        "CodeLlama-70b-hf", "Phind-CodeLlama-34B-v2", "CodeLlama-13b-hf"]},
    "Mistral":  {"base": "#E67E22", "models": [
        "Codestral-22B-v0.1", "Mistral-Small-3.2-24B-Instruct-2506",
        "Magistral-Small-2506", "Mixtral-8x7B-Instruct-v0.1"]},
    "OpenAI":   {"base": "#7D2FB5", "models": [
        "openai-gpt-5.5-2026-04-23", "openai-gpt-4-turbo-2024-04-09",
        "gpt-oss-120b", "gpt-oss-20b"]},
    # The only neutral: with 22 chromatic colours the hue wheel is full, and a
    # single achromatic series stays distinct under any colour-vision deficiency.
    "Gemini":   {"base": "#2F2F2F", "models": ["gemini-gemini-3.5-flash"]},
    "StarCoder":{"base": "#A81070", "models": ["starcoder2-15b"]},
}

# Markers restart per family; combined with the shade ramp they keep
# same-family lines distinguishable.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">", "<", "p"]

LABEL_OVERRIDES = {
    "openai-gpt-5.5-2026-04-23":  "GPT-5.5",
    "openai-gpt-4-turbo-2024-04-09": "GPT-4-turbo",
    "gemini-gemini-3.5-flash":    "Gemini-3.5-flash",
    "deepseek-coder-6.7b-base":   "DeepSeek-Coder-6.7B-base",
    "deepseek-coder-6.7b-instruct": "DeepSeek-Coder-6.7B",
    "starcoder2-15b":             "StarCoder2-15B",
}


def _linear(rgb):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in rgb]


def _contrast_white(rgb) -> float:
    r, g, b = _linear(rgb)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 1.019 / (lum + 0.05)


def _oklab_L(rgb) -> float:
    """Perceptual lightness (OKLab L). HLS lightness is not perceptual:
    at equal HLS l, greens/cyans/yellows read far lighter than blues."""
    r, g, b = _linear(rgb)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s


def _search_l(h: float, s: float, pred) -> float:
    """Largest HLS lightness in [0.12, 0.9] whose colour still satisfies pred."""
    lo, hi = 0.12, 0.9
    for _ in range(20):
        mid = (lo + hi) / 2
        if pred(colorsys.hls_to_rgb(h, mid, s)):
            lo = mid
        else:
            hi = mid
    return lo


def _shades(base_hex: str, n: int) -> list[tuple]:
    """n variants of a base colour, darkest first.

    Members run from a dark anchor to the family's contrast-safe light end,
    evenly spaced in OKLab lightness, with a slight hue rotation. n == 1
    returns the base colour unchanged.
    """
    r, g, b = mcolors.to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n == 1:
        return [(r, g, b)]
    hues = (h + np.linspace(-0.025, 0.025, n)) % 1.0
    sats = np.minimum(1.0, np.linspace(s * 1.05, s * 1.25, n))
    # light end capped by contrast on white, evaluated at its own rotated hue
    l_light = _search_l(float(hues[-1]), float(sats[-1]),
                        lambda c: _contrast_white(c) >= 2.1)
    L_hi = _oklab_L(colorsys.hls_to_rgb(float(hues[-1]), l_light, float(sats[-1])))
    targets = np.linspace(0.47, min(L_hi, 0.765), n)
    colors = []
    for hi_, si, target in zip(hues, sats, targets):
        li = _search_l(float(hi_), float(si), lambda c: _oklab_L(c) <= target)
        colors.append(colorsys.hls_to_rgb(float(hi_), li, float(si)))
    return colors


def _build_registry() -> dict[str, dict]:
    reg: dict[str, dict] = {}
    for fam in FAMILIES.values():
        models = fam["models"]
        for i, (model, color) in enumerate(zip(models, _shades(fam["base"], len(models)))):
            reg[model] = {"color": color, "marker": _MARKERS[i % len(_MARKERS)]}
    return reg


MODEL_STYLE = _build_registry()

_UNKNOWN_WARNED: set[str] = set()
_FALLBACK_COLORS = ["#b3b3b3", "#8d8d8d", "#c9c9c9", "#9e9e9e", "#bcbcbc"]


def _fallback(model: str) -> dict:
    if model not in _UNKNOWN_WARNED:
        print(f"[plotstyle] WARNING: '{model}' not in registry — using grey. "
              f"Add it to FAMILIES in plotstyle.py for a stable colour.")
        _UNKNOWN_WARNED.add(model)
    idx = len(_UNKNOWN_WARNED) - 1
    return {"color": _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)],
            "marker": _MARKERS[idx % len(_MARKERS)]}


def _style_for(model: str) -> dict:
    # dict.get(model, _fallback(model)) would evaluate the fallback (and warn)
    # even for known models.
    return MODEL_STYLE[model] if model in MODEL_STYLE else _fallback(model)


def color_for(model: str):
    return _style_for(model)["color"]


def marker_for(model: str) -> str:
    return _style_for(model)["marker"]


def family_of(model: str) -> str:
    for name, fam in FAMILIES.items():
        if model in fam["models"]:
            return name
    return "Other"


def _auto_short(name: str) -> str:
    name = re.sub(r"-Instruct(?:-\d{4})?$", "", name)
    name = re.sub(r"-v\d+\.\d+$", "", name)
    return name


def label_for(model: str) -> str:
    return LABEL_OVERRIDES.get(model, _auto_short(model))


# ── Global rcParams ───────────────────────────────────────────────────────────

def apply_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        # Embed real (Type-42) fonts so PDF text stays editable/searchable.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


apply_rcparams()

# ── Axis and line styling ─────────────────────────────────────────────────────

GRID_COLOR = "#cccccc"
REF_COLOR = "#909090"   # ideal / reference lines (lighter than the charcoal series)

# White marker edges keep markers legible where many lines cross.
LINE_KW = dict(linewidth=1.5, markersize=6,
               markeredgecolor="white", markeredgewidth=0.7)


def style_axis(ax, *, xgrid=False, ygrid=True, pct_axis="y"):
    """Consistent spines, grid, and optional percentage-formatted axis.

    pct_axis: "y", "x", or None. On a percentage axis ticks snap to 20% steps
    and get a % formatter.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        if pct_axis == "y":
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=0.6, zorder=0)
    if xgrid:
        if pct_axis == "x":
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="x", which="major", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if pct_axis == "y":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    elif pct_axis == "x":
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(labelsize=9)


# ── Saving ────────────────────────────────────────────────────────────────────

FORMATS = ("pdf", "png")
_DPI = 300


def save(fig, output_dir, base_fname, formats=FORMATS):
    """Save `fig` as base_fname.<ext> for each format (or show if no output_dir)."""
    if not output_dir:
        plt.show()
        plt.close(fig)
        return
    os.makedirs(output_dir, exist_ok=True)
    for ext in formats:
        path = os.path.join(output_dir, f"{base_fname}.{ext}")
        fig.savefig(path, dpi=_DPI)
        print(f"Saved: {path}")
    plt.close(fig)


def save_both(fig, output_dir, base_fname, formats=FORMATS):
    """Save with titles (<base>_with_title), then strip titles and save (<base>)."""
    if not output_dir:
        plt.show()
        plt.close(fig)
        return
    os.makedirs(output_dir, exist_ok=True)
    for ext in formats:
        path = os.path.join(output_dir, f"{base_fname}_with_title.{ext}")
        fig.savefig(path, dpi=_DPI)
        print(f"Saved: {path}")
    for ax in fig.axes:
        ax.set_title("")
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    fig.tight_layout()
    for ext in formats:
        path = os.path.join(output_dir, f"{base_fname}.{ext}")
        fig.savefig(path, dpi=_DPI)
        print(f"Saved: {path}")
    plt.close(fig)


# ── Model-subset filters (shared by all plot scripts) ─────────────────────────

def is_reasoning(name: str) -> bool:
    return bool(re.search(r"R1|Qwen3|Magistral", name, re.IGNORECASE))


def is_instruct(name: str) -> bool:
    return bool(re.search(r"Instruct|Codestral|Mistral|Mixtral|Magistral|gpt-oss",
                          name, re.IGNORECASE))


PRESETS = {
    "all":          lambda models: models,
    "instruct":     lambda models: [m for m in models if is_instruct(m)],
    "no-reasoning": lambda models: [m for m in models if not is_reasoning(m)],
    "reasoning":    lambda models: [m for m in models if is_reasoning(m)],
    "base":         lambda models: [m for m in models
                                    if not is_instruct(m) and not is_reasoning(m)],
}


# ── Path / IO helpers ─────────────────────────────────────────────────────────

def resolve_metrics_dir(value: str) -> str:
    """Accept either a real directory or a run name under analysis/outputs/."""
    if os.path.isdir(value):
        return value
    candidate = os.path.join(os.path.dirname(__file__), "outputs", value)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"'{value}' is not a valid directory and was not found under outputs/")


def default_output_dir(metrics_dir: str) -> str:
    name = os.path.basename(os.path.normpath(metrics_dir))
    return os.path.join(os.path.dirname(__file__), "plots", name)


def load_csvs(metrics_dir: str, prefix: str) -> pd.DataFrame:
    """Concatenate every <prefix>*.csv in a directory into one DataFrame."""
    frames = []
    for fname in sorted(os.listdir(metrics_dir)):
        if fname.startswith(prefix) and fname.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(metrics_dir, fname)))
    if not frames:
        raise FileNotFoundError(f"No {prefix}*.csv files found in {metrics_dir}")
    return pd.concat(frames, ignore_index=True)


def fname_suffix(args) -> str:
    """Filename suffix recording which model subset was plotted."""
    parts = []
    if getattr(args, "models", None):
        parts.append("custom")
    elif args.filter != "all":
        parts.append(args.filter.replace("-", ""))
    if getattr(args, "exclude", None):
        parts.append("excl")
    top = getattr(args, "top", None)
    if top and not re.fullmatch(r"top\d+", args.filter):
        parts.append(f"top{top}")
    return ("_" + "_".join(parts)) if parts else ""
