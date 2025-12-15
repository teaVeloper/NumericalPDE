from __future__ import annotations

from math import pi, sqrt
from pathlib import Path

from netgen.geom2d import unit_square
from ngsolve import (
    CF,
    H1,
    BilinearForm,
    GridFunction,
    InnerProduct,
    Integrate,
    LinearForm,
    Mesh,
    Preconditioner,
    TaskManager,
    dx,
    grad,
    runner,
    sin,
    sqrt,
    x,
    y,
)

from ..cli import runner
from ..study.studies import run_study
from ..viz.plots import plot_group


@runner("poisson")
def poisson_runner(*, out: Path, plot: str):
    # --- Exact solution (Poisson on unit square) ---
    # Choose u(x,y) = sin(pi x) sin(pi y), then
    # -Δu = 2*pi^2 * sin(pi x) sin(pi y)
    # Choose refinement ladder
    hs = [0.3 / (2**i) for i in range(5)]
    order = 4  # try 1,2,3...

    study = run_study(
        solve_fn=poisson_solve,
        hs=hs,
        orders=order,
        params_list=[{}],  # no extra parameters for Poisson
        errors={"L2": poisson_err_L2, "H1": poisson_err_H1},
        store_solution=True,  # set False if you only want error numbers
    )

    grp = study.group(order=order, params={})

    plot_group(grp, title=f"Poisson convergence on unit square (p={order})")


def poisson_solve(*, h: float, order: int, **params):
    """
    Solve: -Δu = f in Ω, u=0 on ∂Ω
    Returns (gfu, mesh, extra)
    """
    # uex = CF(x * (1 - x) * y * (1 - y))
    uex = CF(sin(pi * x) * sin(pi * y))
    graduex = CF((uex.Diff(x), uex.Diff(y)))
    divgraduex = CF(graduex[0].Diff(x) + graduex[1].Diff(y))
    # 1) Domain & mesh
    ngmesh = unit_square.GenerateMesh(maxh=h)
    mesh = Mesh(ngmesh)

    # 2) FE space (Dirichlet everywhere)
    V = H1(mesh, order=order, dirichlet=".*")

    # 3) Trial/ test
    u, v = V.TrialFunction(), V.TestFunction()

    # 4) Forms
    a = BilinearForm(V, symmetric=True)
    a += grad(u) * grad(v) * dx

    L = LinearForm(V)
    L += -divgraduex * v * dx

    # 5) Solve
    gfu = GridFunction(V)
    pre = Preconditioner(a, "multigrid")  # "direct" also fine for small meshes

    with TaskManager():
        a.Assemble()
        L.Assemble()
        gfu.vec.data = a.mat.Inverse(freedofs=V.FreeDofs()) * L.vec

    # Put any metadata you like into extra (dofs, etc.)
    extra = {"dofs": V.ndof, "uex": uex, "duex": graduex}
    return gfu, mesh, extra


# --- Error evaluators ---
def poisson_err_L2(gfu, mesh, extra):
    uex = extra["uex"]
    return sqrt(Integrate((gfu - uex) ** 2, mesh))


def poisson_err_H1(gfu, mesh, extra):
    uex = extra["uex"]
    graduex = extra["duex"]
    diff = grad(gfu) - graduex
    return float(sqrt(Integrate(InnerProduct(diff, diff), mesh)))
