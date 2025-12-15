from __future__ import annotations

from dataclasses import dataclass
from math import exp as mexp
from math import pi, sqrt
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import ngsolve.webgui as webgui
import numpy as np
from netgen.geom2d import unit_square
from netgen.occ import *
from ngsolve import *
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
    cos,
    dx,
)
from ngsolve import exp as ngexp
from ngsolve import grad, sin, specialcf, sqrt, x, y
from ngsolve.meshes import Make1DMesh

# pick one of the hs you used above
# h_pick = hs[-1]
# rr = grp[h_pick]
# draw_run(rr, name=f"u_h (h={h_pick:g}, p={order})")


def uex_duex_eps(eps: float):
    k = 11 * pi
    denom = 1 + (eps * k) ** 2

    # scalar constants (Python floats!)
    A = eps / denom
    B = -1.0 / (k * denom)

    q = mexp(-1.0 / eps)  # <-- float
    inv_em1 = q / (1.0 - q)  # <-- float

    C2 = -(2.0 / (k * denom)) * inv_em1  # <-- float
    C1 = 1.0 / (k * denom) - C2  # <-- float

    u = C1 + C2 * ngexp(x / eps) + A * sin(k * x) + B * cos(k * x)
    du = (C2 / eps) * ngexp(x / eps) + A * k * cos(k * x) - B * k * sin(k * x)

    return CF(u), CF(du)


def err_L2_advecdiff(gfu, mesh, extra):
    uex = extra["uex"]
    return float(np.sqrt(Integrate((gfu - uex) ** 2, mesh)))


def err_H1_advecdiff(gfu, mesh, extra):
    duex = extra["duex"]
    return float(np.sqrt(Integrate((grad(gfu)[0] - duex) ** 2, mesh)))


def get_solver(eps: float, beta: float = 1.0):
    uex, duex = uex_duex_eps(eps)

    def solve(*, h: float, order: int, **params):
        mesh = Make1DMesh(int(round(1 / h)))
        f = CF(sin(11 * pi * x))
        V = H1(mesh, order=order, dirichlet=".*")
        u, v = V.TnT()

        a = BilinearForm(V, symmetric=False)
        a += eps * InnerProduct(grad(u), grad(v)) * dx
        a += beta * grad(u)[0] * v * dx

        L = LinearForm(V)
        L += f * v * dx

        gfu = GridFunction(V)
        with TaskManager():
            a.Assemble()
            L.Assemble()
            gfu.vec.data = a.mat.Inverse(freedofs=V.FreeDofs()) * L.vec

        extra = {"dofs": V.ndof, "eps": eps, "beta": beta, "uex": uex, "duex": duex}
        return gfu, mesh, extra

    return solve


def get_modified_solver(eps: float, beta: float = 1.0):
    uex, duex = uex_duex_eps(eps)

    def solve(*, h: float, order: int, **params):
        N = max(2, int(round(1 / h)))
        mesh = Make1DMesh(N)
        f = CF(sin(11 * pi * x))
        V = H1(mesh, order=order, dirichlet=".*")
        u, v = V.TnT()

        eps_h = eps + abs(beta) * h / 2

        a = BilinearForm(V, symmetric=False)
        a += eps_h * InnerProduct(grad(u), grad(v)) * dx
        a += beta * grad(u)[0] * v * dx

        L = LinearForm(V)
        L += f * v * dx

        gfu = GridFunction(V)
        with TaskManager():
            a.Assemble()
            L.Assemble()
            gfu.vec.data = a.mat.Inverse(freedofs=V.FreeDofs()) * L.vec

        extra = {
            "dofs": V.ndof,
            "eps": eps,
            "beta": beta,
            "eps_h": eps_h,
            "uex": uex,
            "duex": duex,
        }
        return gfu, mesh, extra

    return solve


eps_list = [10.0 ** (-k) for k in range(0, 13)]  # 1e0 ... 1e-12
hs = [0.2 / (2**i) for i in range(9)]  # 0.2 ... 0.000390625
order = 4

errors = {"L2": err_L2_advecdiff, "H1": err_H1_advecdiff}

all_std: Dict[float, ConvergenceStudy] = {}
all_mod: Dict[float, ConvergenceStudy] = {}

for eps in eps_list:
    std = run_study(
        solve_fn=get_solver(eps, beta=1.0),
        hs=hs,
        orders=order,
        params_list=[{}],
        errors=errors,
        store_solution=True,
    )
    mod = run_study(
        solve_fn=get_modified_solver(eps, beta=1.0),
        hs=hs,
        orders=order,
        params_list=[{}],
        errors=errors,
        store_solution=True,
    )

    all_std[eps] = std
    all_mod[eps] = mod

    grp_std = std.group(order=order, params={})
    grp_mod = mod.group(order=order, params={})

    print(f"\n=== eps={eps:g} | standard ===")
    print(table_group(grp_std))
    plot_group(grp_std, title=f"Standard Galerkin (eps={eps:g}, p={order})")

    print(f"\n=== eps={eps:g} | modified ===")
    print(table_group(grp_mod))
    plot_group(grp_mod, title=f"Modified (eps={eps:g}, p={order})")


eps = 1e-4
plot_compare(
    {"standard": all_std[eps], "modified": all_mod[eps]},
    order=order,
    params={},
    error_name="L2",
    title=f"Compare methods (eps={eps:g}, p={order}) — L2",
    annotate=False,
)

plot_compare(
    {"standard": all_std[eps], "modified": all_mod[eps]},
    order=order,
    params={},
    error_name="H1",
    title=f"Compare methods (eps={eps:g}, p={order}) — H1",
    annotate=False,
)

eps = 1e-4
grp = all_mod[eps].group(order=order, params={})
h_pick = hs[-1]
rr = grp[h_pick]
draw_run(rr, name=f"u_h (modified, eps={eps:g}, h={h_pick:g}, p={order})")


def get_supg_solver(eps: float, beta: float = 1.0):
    uex, duex = uex_duex_eps(eps)

    def fosls_solve(*, h: float, order: int, **params):
        mesh = Make1DMesh(int(round(1 / h)))

        # 1. Use a Compound Space: u in H1, sigma in H1
        # We need H1 for sigma because we differentiate it in the Balance equation
        V_u = H1(mesh, order=order, dirichlet=".*")
        V_sig = H1(mesh, order=order)
        fes = V_u * V_sig

        (u, sig), (v, tau) = fes.TnT()

        # 2. Define the operators for the two equations
        # Eq 1: Flux def (sig - ux = 0)
        res1 = sig - grad(u)[0]
        res1_test = tau - grad(v)[0]

        # Eq 2: PDE (-eps*sig_x + beta*sig = f) note: using sig instead of ux!
        # We use sig.Diff(x) which is fine because sig is in H1
        op2 = -eps * grad(sig)[0] + beta * sig
        op2_test = -eps * grad(tau)[0] + beta * tau

        # 3. LSFEM Bilinear Form (Symmetric!)
        a = BilinearForm(fes, symmetric=True)
        a += (res1 * res1_test) * dx
        a += (op2 * op2_test) * dx

        # 4. Linear Form
        f_func = CF(sin(11 * pi * x))
        L = LinearForm(fes)
        # The source term f appears in the second equation's residual norm
        L += (f_func * op2_test) * dx

        gfu = GridFunction(fes)
        a.Assemble()
        L.Assemble()

        # Solve
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * L.vec

        # Extract just the u component for plotting/error calc
        gfu_u = gfu.components[0]

        return (
            gfu_u,
            mesh,
            {
                "dofs": fes.ndof,
                "uex": uex,
                "duex": duex,
                "eps": eps,
                "beta": beta,
                "tau": tau,
            },
        )

    return fosls_solve


def _get_wrong_solver(eps: float, beta: float = 1.0):
    uex, duex = uex_duex_eps(eps)

    def solve(*, h: float, order: int, **params):
        N = max(2, int(round(1 / h)))
        mesh = Make1DMesh(N)

        f = CF(sin(11 * pi * x))
        V = H1(mesh, order=order, dirichlet=".*")
        u, v = V.TnT()

        ux = u.Diff(x)
        vx = v.Diff(x)
        uxx = ux.Diff(x)

        Lu = -eps * uxx + beta * ux
        v_supg = beta * vx

        hK = specialcf.mesh_size

        # robust tau: harmonic blend of advective and diffusive scalings
        tau_adv = hK / (2 * abs(beta) + 1e-30)
        tau_diff = hK * hK / (12 * eps + 1e-30)
        tau = 1.0 / (1.0 / tau_adv + 1.0 / tau_diff)

        a = BilinearForm(V, symmetric=False)
        a += eps * ux * vx * dx
        a += beta * ux * v * dx
        a += tau * Lu * v_supg * dx

        L = LinearForm(V)
        L += f * v * dx
        L += tau * f * v_supg * dx

        gfu = GridFunction(V)
        a.Assemble()
        L.Assemble()
        gfu.vec.data = a.mat.Inverse(freedofs=V.FreeDofs()) * L.vec

        extra = {
            "eps": float(eps),
            "beta": float(beta),
            "uex": uex,
            "duex": duex,
            "tau": tau,
        }
        return gfu, mesh, extra

    return solve


def get_fosls_solver(eps: float, beta: float = 1.0):
    uex, duex = uex_duex_eps(eps)

    def solve(*, h: float, order: int, **params):
        # 1. Mesh
        mesh = Make1DMesh(int(round(1 / h)))

        # 2. FOSLS requires a System: u (scalar) and sigma (scalar flux)
        # Both must be H1 to allow differentiation
        V_u = H1(mesh, order=order, dirichlet=".*")  # u=0 on boundary
        V_sig = H1(mesh, order=order)  # sigma free
        fes = V_u * V_sig

        (u, sig), (v, tau) = fes.TnT()

        # 3. The Operators
        # Eq 1: Flux Definition:  sig - u_x = 0
        # Eq 2: Balance Law:      -eps*sig_x + beta*sig = f

        u_x = grad(u)[0]
        v_x = grad(v)[0]
        sig_x = grad(sig)[0]
        tau_x = grad(tau)[0]

        # Residual 1 (Flux): sig - u_x
        res1 = sig - u_x
        res1_test = tau - v_x

        # Residual 2 (PDE): -eps*sig_x + beta*sig
        # Note: We use sig_x, avoiding second derivatives of u!
        op2 = -eps * sig_x + beta * sig
        op2_test = -eps * tau_x + beta * tau

        # 4. Bilinear Form (Symmetric Least Squares)
        # a( (u,sig), (v,tau) ) = (L1(u,sig), L1(v,tau)) + (L2(u,sig), L2(v,tau))
        a = BilinearForm(fes, symmetric=True)
        a += (res1 * res1_test) * dx
        a += (op2 * op2_test) * dx

        # 5. Linear Form
        f_func = CF(sin(11 * pi * x))
        L = LinearForm(fes)
        # The source f is part of the second residual
        L += (f_func * op2_test) * dx

        # 6. Solve
        gfu = GridFunction(fes)
        a.Assemble()
        L.Assemble()
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * L.vec

        # 7. Extract Solution
        # gfu.components[0] is u, gfu.components[1] is sigma
        gfu_u = gfu.components[0]

        # We pass 'sigma' out in extra in case we want to plot flux later
        extra = {
            "dofs": fes.ndof,
            "uex": uex,
            "duex": duex,
            "gfu_sig": gfu.components[1],
            "eps": eps,
            "beta": beta,
        }
        return gfu_u, mesh, extra

    return solve


def err_GLS_norm_advecdiff(gfu, mesh, extra):
    eps = float(extra["eps"])
    beta = float(extra["beta"])
    uex = extra["uex"]

    # derivatives
    uh_x = gfu.Diff(x)
    uh_xx = uh_x.Diff(x)

    uex_x = uex.Diff(x)
    uex_xx = uex_x.Diff(x)

    v = gfu - uex
    vx = uh_x - uex_x
    vxx = uh_xx - uex_xx

    # use SAME tau as solver (elementwise)
    tau = extra.get("tau", specialcf.mesh_size / (2 * abs(beta) + 1e-30))

    Lv = -eps * vxx + beta * vx

    val = Integrate(v * v + eps * vx * vx + tau * Lv * Lv, mesh)
    return float(np.sqrt(val))


def err_SUPG_norm(gfu, mesh, extra):
    eps = float(extra["eps"])
    beta = float(extra["beta"])
    uex = extra["uex"]
    tau = extra["tau"]  # This is a CoefficientFunction from the solver

    # --- 1. GridFunction Derivatives ---
    # First derivative (Standard)
    uh_x = gfu.Diff(x)

    # Second derivative (MUST use 'hesse' operator)
    # In 1D, the Hessian is a 1x1 matrix containing u_xx.
    # We select [0] to get it as a scalar.
    uh_xx = gfu.Operator("hesse")[0]

    # --- 2. Exact Solution Derivatives ---
    # uex is symbolic (CF), so we can just chain Diff safely
    u_x = uex.Diff(x)
    u_xx = u_x.Diff(x)

    # --- 3. Residual Calculation ---
    v = gfu - uex
    vx = uh_x - u_x
    vxx = uh_xx - u_xx

    # Calculate the residual of the ERROR (Lv)
    # Strong form: -eps*u_xx + beta*u_x
    Lv = -eps * vxx + beta * vx

    # Integrate the GLS/SUPG norm
    # ||e||^2_GLS = ||e||^2 + eps*||e'||^2 + tau*||Le||^2
    val = Integrate(v * v + eps * vx * vx + tau * Lv * Lv, mesh)

    return float(np.sqrt(val))


def err_fosls_functional(gfu, mesh, extra):
    # Reconstruct the residuals to see how well we solved the system
    eps = 1e-4  # You might want to pass this in extra if it varies
    if "params" in extra:  # Handle parameter passing if needed
        pass

    # Retrieve the sigma component we saved earlier
    gfu_sig = extra["gfu_sig"]
    beta = extra["beta"]

    u_x = grad(gfu)[0]
    sig = gfu_sig
    sig_x = grad(gfu_sig)[0]

    # Exact f for residual calc
    f = CF(sin(11 * pi * x))

    # Recalculate residuals
    res1 = sig - u_x
    res2 = -eps * sig_x + beta * sig - f

    # The functional value J
    val = Integrate(res1 * res1 + res2 * res2, mesh)
    return float(np.sqrt(val))


gls_errors = {
    "L2": err_L2_advecdiff,
    "H1": err_H1_advecdiff,
    "GLS": err_fosls_functional,
}
for eps in eps_list:
    std = run_study(
        solve_fn=get_fosls_solver(eps, beta=1.0),
        hs=hs,
        orders=order,
        params_list=[{}],
        errors=gls_errors,
        store_solution=True,
    )
    grp = std.group(order=order, params={})
    print(f"\n=== eps={eps:g} | gls ===")
    print(table_group(grp))
    plot_group(grp, title=f"Least Square Galerkin (eps={eps:g}, p={order})")
