from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Number = Union[int, float]
Params = Dict[str, Any]

SolveFn = Callable[..., Tuple[Any, Any, Dict[str, Any]]]  # (gfu, mesh, extra)
ErrorFn = Callable[[Any, Any, Dict[str, Any]], float]  # (gfu, mesh, extra) -> float


def compute_rates(errors: Sequence[float], hs: Sequence[float]) -> List[float]:
    """Local rates p_i = log(E_i/E_{i+1}) / log(h_i/h_{i+1})"""
    rates: List[float] = []
    for i in range(len(errors) - 1):
        Ei, Ej = float(errors[i]), float(errors[i + 1])
        hi, hj = float(hs[i]), float(hs[i + 1])
        if Ej == 0 or hi == hj:
            rates.append(float("nan"))
        else:
            rates.append(float(np.log(Ei / Ej) / np.log(hi / hj)))
    return rates


# Data structures


@dataclass(frozen=True)
class RunSpec:
    h: float
    order: int
    params: Tuple[Tuple[str, Any], ...]


@dataclass
class SolutionStore:
    gfu: Any
    mesh: Any
    extra: Dict[str, Any]


@dataclass
class RunResult:
    spec: RunSpec
    errors: Dict[str, float]
    solution: Optional[SolutionStore] = None
