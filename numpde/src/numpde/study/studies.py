from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..core.structures import (
    ErrorFn,
    Number,
    Params,
    RunResult,
    RunSpec,
    SolutionStore,
    SolveFn,
)


def _params_key(params: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(params.items(), key=lambda kv: kv[0]))


def _h_key(h: float, ndigits: int = 12) -> float:
    return round(float(h), ndigits)


class StudyGroup:
    """One fixed (order, params) slice. Index by h: grp[h] -> RunResult"""

    def __init__(
        self,
        *,
        order: int,
        params: Params,
        results: List[RunResult],
        h_ndigits: int = 12,
    ):
        self.order = int(order)
        self.params = dict(params)
        self._pkey = _params_key(self.params)
        self._h_ndigits = h_ndigits

        self._by_h: Dict[float, RunResult] = {}
        for r in results:
            if r.spec.order != self.order or r.spec.params != self._pkey:
                continue
            self._by_h[_h_key(r.spec.h, self._h_ndigits)] = r

        if not self._by_h:
            raise KeyError(f"No results for order={self.order}, params={self.params}")

    def hs(self) -> List[float]:
        return sorted(self._by_h.keys(), reverse=True)  # coarse -> fine

    def __getitem__(self, h: float) -> RunResult:
        hk = _h_key(h, self._h_ndigits)
        if hk not in self._by_h:
            raise KeyError(f"h={h} not found. Available: {self.hs()}")
        return self._by_h[hk]

    def error_names(self) -> List[str]:
        rr0 = self._by_h[self.hs()[0]]
        return list(rr0.errors.keys())

    def hs_eff(self):
        return [self._by_h[h].solution.extra.get("h_eff", h) for h in self.hs()]


class ConvergenceStudy:
    def __init__(self, *, h_ndigits: int = 12):
        self._results: List[RunResult] = []
        self._h_ndigits = h_ndigits

    @property
    def results(self) -> List[RunResult]:
        return list(self._results)

    def add(self, rr: RunResult) -> None:
        self._results.append(rr)

    def orders(self) -> List[int]:
        return sorted(set(r.spec.order for r in self._results))

    def params_keys(self) -> List[Tuple[Tuple[str, Any], ...]]:
        return sorted(set(r.spec.params for r in self._results))

    def group(self, *, order: int, params: Optional[Params] = None) -> StudyGroup:
        return StudyGroup(
            order=order,
            params=params or {},
            results=self._results,
            h_ndigits=self._h_ndigits,
        )


def run_study(
    *,
    solve_fn: SolveFn,
    hs: Sequence[Number],
    orders: Union[int, Sequence[int]] = 1,
    params_list: Optional[Sequence[Params]] = None,
    errors: Dict[str, ErrorFn],
    store_solution: bool = False,
) -> ConvergenceStudy:
    hs_list = [float(v) for v in hs]
    orders_list = [int(orders)] if isinstance(orders, int) else [int(v) for v in orders]
    params_list = list(params_list) if params_list is not None else [dict()]

    study = ConvergenceStudy()

    for p in params_list:
        pkey = _params_key(p)
        for order in orders_list:
            for h in hs_list:
                gfu, mesh, extra = solve_fn(h=h, order=order, **p)

                err_vals: Dict[str, float] = {}
                for name, fn in errors.items():
                    err_vals[name] = float(fn(gfu, mesh, extra))

                sol = (
                    SolutionStore(gfu=gfu, mesh=mesh, extra=dict(extra))
                    if store_solution
                    else None
                )

                study.add(
                    RunResult(
                        spec=RunSpec(h=float(h), order=int(order), params=pkey),
                        errors=err_vals,
                        solution=sol,
                    )
                )

    return study
