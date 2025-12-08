# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 18

# %% [markdown]
#
# We consider the function
# $
# g(r,\theta) = r^{\tfrac{2}{3}} \,\sin\!\left(\tfrac{2}{3}\theta\right).
# $
# Polar coordinates are defined by
# $
# r = \sqrt{x^2 + y^2}, \qquad \theta = \operatorname{atan2}(y,x).
# $
#
# The domain is the L-shaped region
# $
# \Omega = (-1,1)^2 \setminus \big( (-1,0] \times (-1,0] \big).
# $
#
# We study the Dirichlet problem
#
# \begin{aligned}
# - \Delta u &= 0 && \text{in } \Omega, \\
# u &= g && \text{on } \partial\Omega .
# \end{aligned}
#
#
# Since $ (0,0) \notin \Omega $, we always have $(r>0)$ in the interior, and therefore we may use polar coordinates safely.
#

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# %matplotlib inline

fig, ax = plt.subplots(figsize=(4, 4))

# Big square: (-1,-1) to (1,1)
outer = patches.Rectangle((-1, -1), 2, 2,
                          facecolor="lightgray", edgecolor="black")
ax.add_patch(outer)

# Cut-out lower-left square: (-1,-1) to (0,0)
hole = patches.Rectangle((-1, -1), 1, 1,
                         facecolor="white", edgecolor="black")
ax.add_patch(hole)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal", "box")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("L-shaped domain $\\Omega$")

plt.grid(True, linestyle=":", linewidth=0.5)
plt.show()


# %% [markdown]
# ## Harmonicity of $g$ via the polar Laplacian
#
# The Laplacian in polar coordinates is  
# $$
# \Delta g
# = g_{rr} + \frac{1}{r}\, g_r + \frac{1}{r^2}\, g_{\theta\theta}.
# $$
#
# We have  
# $$
# g(r,\theta) = r^{\frac{2}{3}} \sin\!\left(\frac{2}{3}\theta\right).
# $$
#
# ###  derivatives
#
# $$
# g_r = \frac{2}{3} r^{-\frac{1}{3}} \sin\!\left(\frac{2}{3}\theta\right),
# $$
# $$
# g_{rr}
# = -\frac{2}{9} r^{-\frac{4}{3}} \sin\!\left(\frac{2}{3}\theta\right).
# $$
# $$
# g_\theta = \frac{2}{3} r^{\frac{2}{3}} \cos\!\left(\frac{2}{3}\theta\right),
# $$
# $$
# g_{\theta\theta}
# = -\frac{4}{9} r^{\frac{2}{3}} \sin\!\left(\frac{2}{3}\theta\right).
# $$
#
# ### Plugging into the polar Laplacian
#
# $$
# \begin{aligned}
# \Delta g
# &= -\frac{2}{9} r^{-\frac{4}{3}} \sin\!\left(\frac{2}{3}\theta\right)
#    + \frac{2}{3} r^{-\frac{4}{3}} \sin\!\left(\frac{2}{3}\theta\right)
#    - \frac{4}{9} r^{-\frac{4}{3}} \sin\!\left(\frac{2}{3}\theta\right) \\[0.4em]
# &= \left( -\frac{2}{9} + \frac{6}{9} - \frac{4}{9} \right)
#    r^{-\frac{4}{3}} \sin\!\left(\frac{2}{3}\theta\right)
# = 0.
# \end{aligned}
# $$
#
# Thus $g$ is harmonic on $\Omega$.
#

# %%
# load Netgen/NGSolve 
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import * 

# %%
outer = MoveTo(-1,-1).Rectangle(2,2).Face()
outer.edges.name="outer"
outer.edges.Max(X).name = "r"
outer.edges.Min(X).name = "l"
outer.edges.Min(Y).name = "b"
outer.edges.Max(Y).name = "t"

inner = MoveTo(-1,-1).Rectangle(1,1).Face()
inner.edges.name="interface"


# L-shaped domain: outer minus inner
lshape = outer - inner
lshape.faces.name = "domain"
lshape.edges.name = "boundary"


inner.faces.name="inner"
inner.faces.col = (1, 1, 1)
outer.faces.name="outer"

geo = OCCGeometry(lshape)

#geo = Glue([inner, outer])
Draw(lshape);

# %%
mesh = Mesh(OCCGeometry(lshape, dim=2).GenerateMesh(maxh=0.2))
Draw(mesh);

# %%
mesh.nv, mesh.ne

# %%
fes = H1(mesh, order=6, dirichlet="boundary")
print ("number of dofs =", fes.ndof)

# %%
u = fes.TrialFunction()
v = fes.TestFunction()
a = BilinearForm(grad(u)*grad(v)*dx)

# %%
f = LinearForm(fes)
# no term to add, since RHS is 0
f.Assemble()
a.Assemble()

# %%
from ngsolve import x, y, sqrt, atan2, sin

r     = sqrt(x*x + y*y)
theta = atan2(y, x)
g_exact = r**(2/3) * sin(2*theta/3)


# %%
gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
gfu.Set(g_exact, definedon=mesh.Boundaries("boundary"))

# %%
def solve_equation(h, order):
    mesh = Mesh(OCCGeometry(lshape, dim=2).GenerateMesh(maxh=h))
    fes = H1(mesh, order=order, dirichlet="boundary")
    u = fes.TrialFunction()
    v = fes.TestFunction()
    a = BilinearForm(grad(u)*grad(v)*dx)
    f = LinearForm(fes)
    # no term to add, since RHS is 0
    f.Assemble()
    a.Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
    gfu.Set(g_exact, definedon=mesh.Boundaries("boundary"))
    return gfu, mesh

# %%

Draw (gfu, euler_angles=[-60,-10,-20], deformation=True, scale=0.3);

# %%
Draw (grad(gfu), mesh, order=3, vectors = True);

# %%
Draw(gfu, mesh, "uh")

# %%
# interpolation
uex = CF( g_exact) 
graduex = CF( (uex.Diff(x), uex.Diff(y) ))

errL2 = []
errH1 = []
h = []

for i in range(4):
    h.append( 0.2 / ( 2**i) )
    mesh = Mesh(unit_square.GenerateMesh(maxh=h[-1]))
    fes = H1(mesh, order = 1)
    gfu = GridFunction(fes)
    gfu.Set(uex, dual = True)
    errL2.append (sqrt( Integrate( (gfu - uex)**2, mesh)))
    errH1.append( sqrt( Integrate( (grad(gfu) - graduex)**2, mesh)))

import numpy as np

def compute_rates(err_vec, nel_vec):
    rates = []
    for i in range(len(nel_vec)-1):
        r = np.log(err_vec[i] / err_vec[i+1]) / np.log(h[i] / h[i+1])
        rates.append(r)
    return rates

rates_L2 = compute_rates(errL2, h)
rates_H1 = compute_rates(errH1, h)

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))

# Plot errors
plt.loglog(h, errL2, marker="o", linestyle="--", label="L2 error")
plt.loglog(h, errH1, marker="s", linestyle="-.", label="H1 error")


# Annotate convergence rates
def annotate_rates(nel_vec, err_vec, rates, y_offset=0):
    for i in range(len(rates)):
        x_mid = np.sqrt(nel_vec[i] * nel_vec[i+1])
        y_mid = np.sqrt(err_vec[i] * err_vec[i+1])
        plt.text(x_mid, y_mid*(1+y_offset), f"{rates[i]:.2f}", fontsize=9, ha="center")

annotate_rates(h, errL2, rates_L2, y_offset=0.05)
annotate_rates(h, errH1, rates_H1, y_offset=0.05)


plt.xlabel("mesh size")
plt.ylabel("Error")
plt.title("Error Convergence with Local Rates")
plt.legend()
plt.grid(True, which="both")

plt.show()

print("h\t\tL2 error\t\tL2 rate\t\tH1 error\t\tH1 rate")
print("-"*80)
for i in range(len(h)):
    h_i = h[i]
    eL2 = errL2[i]
    eH1 = errH1[i]
    if i == 0:
        rL2 = ""
        rH1 = ""
    else:
        rL2 = f"{rates_L2[i-1]:.3f}"
        rH1 = f"{rates_H1[i-1]:.3f}"
    print(f"{h_i:.5f}\t{eL2:.5e}\t{rL2:>6}\t\t{eH1:.5e}\t{rH1:>6}")

# %%
h = []
errL2 = []
errH1 = []

for i in range(4):
    h.append( 0.2 / ( 2**i) )
    gfu, mesh = solve_equation(h[-1], 4 )
    eL2 = sqrt( Integrate( (gfu - uex)**2, mesh))
    eH1 = sqrt( Integrate( (grad(gfu) - graduex)**2, mesh))
    errL2.append (eL2)
    errH1.append( eH1)

import numpy as np

def compute_rates(err_vec, nel_vec):
    rates = []
    for i in range(len(nel_vec)-1):
        r = np.log(err_vec[i] / err_vec[i+1]) / np.log(h[i] / h[i+1])
        rates.append(r)
    return rates

rates_L2 = compute_rates(errL2, h)
rates_H1 = compute_rates(errH1, h)

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))

# Plot errors
plt.loglog(h, errL2, marker="o", linestyle="--", label="L2 error")
plt.loglog(h, errH1, marker="s", linestyle="-.", label="H1 error")


# Annotate convergence rates
def annotate_rates(nel_vec, err_vec, rates, y_offset=0):
    for i in range(len(rates)):
        x_mid = np.sqrt(nel_vec[i] * nel_vec[i+1])
        y_mid = np.sqrt(err_vec[i] * err_vec[i+1])
        plt.text(x_mid, y_mid*(1+y_offset), f"{rates[i]:.2f}", fontsize=9, ha="center")

annotate_rates(h, errL2, rates_L2, y_offset=0.05)
annotate_rates(h, errH1, rates_H1, y_offset=0.05)


plt.xlabel("mesh size")
plt.ylabel("Error")
plt.title("Error Convergence with Local Rates")
plt.legend()
plt.grid(True, which="both")

plt.show()

print("h\t\tL2 error\t\tL2 rate\t\tH1 error\t\tH1 rate")
print("-"*80)
for i in range(len(h)):
    h_i = h[i]
    eL2 = errL2[i]
    eH1 = errH1[i]
    if i == 0:
        rL2 = ""
        rH1 = ""
    else:
        rL2 = f"{rates_L2[i-1]:.3f}"
        rH1 = f"{rates_H1[i-1]:.3f}"
    print(f"{h_i:.5f}\t{eL2:.5e}\t{rL2:>6}\t\t{eH1:.5e}\t{rH1:>6}")

# %% [markdown]
# # Exercise 19

# %% [markdown]
#
# We consider the function
# $$
# g(r,\theta) = \sqrt{r}\,\sin\!\left(\frac{\theta}{2}\right),
# $$
# with polar coordinates
# $$
# r = \sqrt{x^2 + y^2}, \qquad \theta = \operatorname{atan2}(y,x).
# $$
#
# The domain is the **upper half-disk**
# $$
# \Omega = \{ (x,y)\in\mathbb{R}^2 : x^2 + y^2 < 1,\; y > 0 \}.
# $$
#
# The boundary is split into:
# - **Neumann part**  
#   $$\Gamma_N = \{ (x,y)\in\partial\Omega : x < 0,\; y = 0 \}$$  
#   (left half of the diameter),
#
# - **Dirichlet part**  
#   $$\Gamma_D = \{ (x,y)\in\partial\Omega : x > 0,\; y = 0 \}$$  
#   (right half of the diameter),
#   $$\Gamma_C = \{ x^2 + y^2 = 1,\; y > 0 \},$$  
#
# The mixed boundary value problem is:
# $$
# \begin{aligned}
# -\Delta u &= 0 &&\text{in } \Omega,\\[0.2em]
# u &= g &&\text{on } \Gamma_D \cup \Gamma_C,\\[0.2em]
# \frac{\partial u}{\partial n} &= 0 &&\text{on } \Gamma_N.
# \end{aligned}
# $$
#

# %% [markdown]
# ## Harmonicity of $g$ on $\Omega$
#
# We use the polar Laplacian:
# $$
# \Delta g = g_{rr} + \frac{1}{r} g_r + \frac{1}{r^2} g_{\theta\theta}.
# $$
#
# With  
# $$
# g(r,\theta) = r^{1/2}\,\sin(\theta/2),
# $$
# Derivatives
# $$
# g_r = \frac{1}{2} r^{-1/2} \sin(\theta/2),
# $$
#
# $$
# g_{rr} = -\frac{1}{4} r^{-3/2} \sin(\theta/2).
# $$
#
# $$
# g_\theta = r^{1/2}\,\frac{1}{2}\cos(\theta/2),
# $$
#
# $$
# g_{\theta\theta}
# = r^{1/2}\,\left(-\frac{1}{4}\sin(\theta/2)\right)
# = -\frac{1}{4} r^{1/2} \sin(\theta/2).
# $$
#
# ### Plugging into the polar Laplacian
# $$
# \begin{aligned}
# \Delta g 
# &= -\frac{1}{4} r^{-3/2}\sin(\theta/2)
#    + \frac{1}{r}\cdot \frac{1}{2} r^{-1/2} \sin(\theta/2)
#    + \frac{1}{r^2} \cdot \left(-\frac{1}{4} r^{1/2} \sin(\theta/2)\right) \\[0.4em]
# &= \left( -\frac14 + \frac12 - \frac14 \right) r^{-3/2} \sin(\theta/2) \\[0.4em]
# &= 0.
# \end{aligned}
# $$
#
# Thus $g$ is harmonic in $\Omega$ (note $r>0$ everywhere in the domain).
#

# %% [markdown]
# ## Normal derivative on the Neumann boundary
#
# On the **Neumann boundary**
# $$
# \Gamma_N = \{ (x,0) : -1 \le x < 0 \},
# $$
# the outward normal is the **negative $y$-direction**:
# $$
# n = (0,-1).
# $$
#
# We compute the gradient of $g$ in Cartesian coordinates:
# $$
# \nabla g = (g_x, g_y).
# $$
#
# On the diameter $y=0$ we have $\theta = 0$ or $\theta = \pi$.  
# On $\Gamma_N$ (left half), $\theta = \pi$.
#
# Since
# $$
# g(r,\theta) = r^{1/2}\sin(\theta/2),
# $$
# we have
# $$
# g(r,\pi) = r^{1/2}\sin(\pi/2) = r^{1/2}.
# $$
#
# Differentiate:
# $$
# g_y = \frac{\partial g}{\partial r}\frac{\partial r}{\partial y} 
#      + \frac{\partial g}{\partial \theta}\frac{\partial \theta}{\partial y}.
# $$
#
# At $y=0$ and $\theta=\pi$:
# - $\dfrac{\partial r}{\partial y} = \dfrac{y}{r} = 0$  
# - $\dfrac{\partial \theta}{\partial y} = \dfrac{x}{r^2}$  
# - but $g_\theta = r^{1/2}\frac12\cos(\theta/2)$  
#   and $\cos(\pi/2)=0$
#
# So:
# $$
# g_y = 0,
# $$
# hence
# $$
# \nabla g \cdot n = (g_x, g_y)\cdot(0,-1) = -g_y = 0.
# $$
#
# Thus the Neumann boundary condition is satisfied identically.
#

# %%
from netgen.occ import *
from netgen.webgui import Draw

# Points
pL   = Pnt(-1, 0, 0)    # left end of diameter
p0   = Pnt( 0, 0, 0)    # midpoint of diameter
pR   = Pnt( 1, 0, 0)    # right end of diameter
pTop = Pnt( 0, 1, 0)    # top of semicircle

# Construct edges
eN_raw = Segment(pL, p0)                # intended Neumann
eD_raw = Segment(p0, pR)                # intended Dirichlet (bottom right)
arc_raw = ArcOfCircle(pR, pTop, pL)     # intended Dirichlet (arc)

w = Wire([eN_raw, arc_raw, eD_raw])
half_circle = Face(w)
half_circle.faces.name = "domain"

# Now classify the *actual* edges of half_circle
for e in half_circle.edges:
    # take a representative point on the edge
    p = e.vertices[0].p
    x, y, z = p

    if abs(y) < 1e-12 and x < 0:
        # bottom-left: Neumann
        e.name = "neumann"
        e.col  = (0, 0, 1)   # blue
    else:
        # bottom-right and arc: Dirichlet
        e.name = "dirichlet"
        e.col  = (1, 0, 0)   # red

Draw(half_circle)


# %%
for e in half_circle.edges:
    p = e.vertices[0].p
    print(e.name, p)


# %%

geo  = OCCGeometry(half_circle, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=0.15))

print(mesh.GetBoundaries())

fes = H1(mesh, order=5, dirichlet="dirichlet")

fd = fes.FreeDofs()           # bit array of length fes.ndof
free = sum(1 for b in fd if b)
print("ndof          =", fes.ndof)
print("free dofs     =", free)
print("dirichlet dofs =", fes.ndof - free)



# %%
Draw(mesh);

# %%
print(mesh.nv)
print(mesh.ne)
print(mesh.GetBoundaries())

# %%


print ("number of dofs =", fes.ndof)



# %%
from ngsolve import x, y, sqrt, atan2, sin

u = fes.TrialFunction()
v = fes.TestFunction()

a = BilinearForm(fes, symmetric=True)
a += grad(u)*grad(v)*dx

f = LinearForm(fes)   # RHS 0
a.Assemble()
f.Assemble()

# exact solution
r     = sqrt(x*x + y*y)
theta = atan2(y, x)
g_exact = sqrt(r) * sin(theta/2)

gfu = GridFunction(fes)
gfu.Set(g_exact, definedon=mesh.Boundaries("dirichlet"))
import numpy as np

vals = np.array(gfu.vec)
print("min =", vals.min(), "max =", vals.max())



# %%
Draw (gfu, euler_angles=[-60,-10,-20], deformation=True, scale=0.3);

# %%
Draw (grad(gfu), mesh, order=3, vectors = True);

# %%
rhs = f.vec - a.mat * gfu.vec
inv = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")

du = gfu.vec.CreateVector()
du.data = inv * rhs
gfu.vec.data += du

vals = np.array(gfu.vec)
print("u_h min/max:", vals.min(), vals.max(), "  norm:", gfu.vec.Norm())
Draw(gfu, mesh, "u_h", deformation=True, scale=0.3)


# %%
Draw(g_exact, mesh, "u", deforemation=True, scale=0.3)

# %%

uex = CF( g_exact) 
graduex = CF( (uex.Diff(x), uex.Diff(y) ))

errL2 = []
errH1 = []
h = []

for i in range(4):
    h.append( 0.2 / ( 2**i) )
    mesh = Mesh(unit_square.GenerateMesh(maxh=h[-1]))
    fes = H1(mesh, order = 1)
    gfu = GridFunction(fes)
    gfu.Set(uex, dual = True)
    errL2.append (sqrt( Integrate( (gfu - uex)**2, mesh)))
    errH1.append( sqrt( Integrate( (grad(gfu) - graduex)**2, mesh)))

import numpy as np

def compute_rates(err_vec, nel_vec):
    rates = []
    for i in range(len(nel_vec)-1):
        r = np.log(err_vec[i] / err_vec[i+1]) / np.log(h[i] / h[i+1])
        rates.append(r)
    return rates

rates_L2 = compute_rates(errL2, h)
rates_H1 = compute_rates(errH1, h)

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))

# Plot errors
plt.loglog(h, errL2, marker="o", linestyle="--", label="L2 error")
plt.loglog(h, errH1, marker="s", linestyle="-.", label="H1 error")


# Annotate convergence rates
def annotate_rates(nel_vec, err_vec, rates, y_offset=0):
    for i in range(len(rates)):
        x_mid = np.sqrt(nel_vec[i] * nel_vec[i+1])
        y_mid = np.sqrt(err_vec[i] * err_vec[i+1])
        plt.text(x_mid, y_mid*(1+y_offset), f"{rates[i]:.2f}", fontsize=9, ha="center")

annotate_rates(h, errL2, rates_L2, y_offset=0.05)
annotate_rates(h, errH1, rates_H1, y_offset=0.05)


plt.xlabel("mesh size")
plt.ylabel("Error")
plt.title("Error Convergence with Local Rates")
plt.legend()
plt.grid(True, which="both")

plt.show()

print("h\t\tL2 error\t\tL2 rate\t\tH1 error\t\tH1 rate")
print("-"*80)
for i in range(len(h)):
    h_i = h[i]
    eL2 = errL2[i]
    eH1 = errH1[i]
    if i == 0:
        rL2 = ""
        rH1 = ""
    else:
        rL2 = f"{rates_L2[i-1]:.3f}"
        rH1 = f"{rates_H1[i-1]:.3f}"
    print(f"{h_i:.5f}\t{eL2:.5e}\t{rL2:>6}\t\t{eH1:.5e}\t{rH1:>6}")


# %%
