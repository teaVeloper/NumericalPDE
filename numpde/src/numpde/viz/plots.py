from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ..core.structures import Params, RunResult, compute_rates
from ..study.studies import ConvergenceStudy, StudyGroup


def _annotate_rates(ax, hs, errs, rates, y_offset=0.05):
    for i, r in enumerate(rates):
        x_mid = np.sqrt(hs[i] * hs[i + 1])
        y_mid = np.sqrt(errs[i] * errs[i + 1])
        ax.text(x_mid, y_mid * (1 + y_offset), f"{r:.2f}", fontsize=9, ha="center")


def table_group(grp: StudyGroup, *, which: Optional[Sequence[str]] = None) -> str:
    hs = grp.hs()
    names = grp.error_names()
    if which is not None:
        wanted = set(which)
        names = [n for n in names if n in wanted]

    cols = []
    for name in names:
        vals = [grp[h].errors[name] for h in hs]
        cols.append((name, vals, compute_rates(vals, hs)))

    header = f"order={grp.order}"
    if grp.params:
        header += " | " + ", ".join(f"{k}={v}" for k, v in sorted(grp.params.items()))

    lines = [header]
    lines.append("h\t\t" + "\t\t".join([f"{n} err\t{n} rate" for n in names]))
    lines.append("-" * (16 + 24 * len(names)))

    for i, h in enumerate(hs):
        row = [f"{h:.5e}"]
        for name, vals, rates in cols:
            err = vals[i]
            rate = "" if i == 0 else f"{rates[i-1]:.3f}"
            row.append(f"{err:.5e}\t{rate:>8}")
        lines.append("\t".join(row))

    return "\n".join(lines)


def plot_group(
    grp: StudyGroup,
    *,
    title: Optional[str] = None,
    which: Optional[Sequence[str]] = None,
    annotate: bool = True,
):
    hs = grp.hs()
    names = grp.error_names()
    if which is not None:
        wanted = set(which)
        names = [n for n in names if n in wanted]

    fig, ax = plt.subplots(figsize=(9, 6))

    for name in names:
        ys = [grp[h].errors[name] for h in hs]
        ax.loglog(hs, ys, marker="o", linestyle="--", label=name)
        if annotate and len(hs) >= 2:
            _annotate_rates(ax, hs, ys, compute_rates(ys, hs), y_offset=0.05)

    suffix = f"p={grp.order}"
    if grp.params:
        suffix += ", " + ", ".join(f"{k}={v}" for k, v in sorted(grp.params.items()))

    ax.set_xlabel("mesh size h")
    ax.set_ylabel("Error")
    ax.set_title(title or f"Error convergence ({suffix})")
    ax.grid(True, which="both")
    ax.legend()
    plt.show()


def plot_compare(
    studies: Dict[str, ConvergenceStudy],
    *,
    order: int,
    params: Params,
    error_name: str,
    title: str,
    annotate: bool = False,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, study in studies.items():
        grp = study.group(order=order, params=params)
        hs = grp.hs()
        ys = [grp[h].errors[error_name] for h in hs]

        ax.loglog(hs, ys, marker="o", linestyle="--", label=label)
        if annotate and len(hs) >= 2:
            _annotate_rates(ax, hs, ys, compute_rates(ys, hs), y_offset=0.05)

    ax.set_xlabel("mesh size h")
    ax.set_ylabel(error_name)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    plt.show()


def draw_run(rr: RunResult, *, name: str = "u_h", **draw_kwargs):
    if rr.solution is None:
        raise ValueError("No stored solution. Re-run with store_solution=True.")
    return webgui.Draw(rr.solution.gfu, name=name, **draw_kwargs)
