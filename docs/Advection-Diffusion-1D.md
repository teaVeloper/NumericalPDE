# Exact Solution Derivation
We consider the 1D advection-diffusion problem:
$$
-\varepsilon u''(x) + u'(x) = \sin(k x) \quad \text{on } \Omega=(0,1)
$$
with boundary conditions $u(0) = u(1) = 0$ and the constant $k = 11\pi$.

## Step 1: Homogeneous Solution
First, solve the homogeneous equation $-\varepsilon u_h'' + u_h' = 0$.
The characteristic equation is:
$$
-\varepsilon r^2 + r = 0 \implies r(1 - \varepsilon r) = 0
$$
The roots are $r_1 = 0$ and $r_2 = \frac{1}{\varepsilon}$. Thus, the homogeneous solution is:
$$
u_h(x) = C_1 + C_2 e^{x/\varepsilon}
$$

## Step 2: Particular Solution
We seek a particular solution $u_p(x)$ for the source term $f(x) = \sin(kx)$. We use the ansatz:
$$
u_p(x) = A \sin(kx) + B \cos(kx)
$$
Calculating the derivatives:
$$
\begin{aligned}
u_p'(x) &= kA \cos(kx) - kB \sin(kx) \\
u_p''(x) &= -k^2 A \sin(kx) - k^2 B \cos(kx)
\end{aligned}
$$
Substitute these into the PDE $-\varepsilon u'' + u' = \sin(kx)$:
$$
-\varepsilon \left( -k^2 A \sin(kx) - k^2 B \cos(kx) \right) + \left( kA \cos(kx) - kB \sin(kx) \right) = \sin(kx)
$$
Grouping coefficients for $\sin(kx)$ and $\cos(kx)$:
$$
\sin(kx) \left[ \varepsilon k^2 A - kB \right] + \cos(kx) \left[ \varepsilon k^2 B + kA \right] = 1 \cdot \sin(kx)
$$
Comparing coefficients yields the linear system:
1.  $\varepsilon k^2 B + kA = 0 \implies A = -\varepsilon k B$
2.  $\varepsilon k^2 A - kB = 1$

Substitute (1) into (2):
$$
\varepsilon k^2 (-\varepsilon k B) - kB = 1 \\
-B (\varepsilon^2 k^3 + k) = 1
$$
Solving for the constants:
$$
B = -\frac{1}{k(1 + \varepsilon^2 k^2)}, \qquad A = \frac{\varepsilon}{1 + \varepsilon^2 k^2}
$$
Let $D = 1 + \varepsilon^2 k^2$. The particular solution is:
$$
u_p(x) = \frac{1}{kD} \left( \varepsilon k \sin(kx) - \cos(kx) \right)
$$

## Step 3: Impose Boundary Conditions
The general solution is $u(x) = C_1 + C_2 e^{x/\varepsilon} + u_p(x)$.

**At x = 0:**
$$u(0) = C_1 + C_2 + u_p(0) = 0$$
Using $u_p(0) = B = -\frac{1}{kD}$:
$$C_1 + C_2 = \frac{1}{kD} \quad \text{(Eq. I)}$$

**At x = 1:**
$$u(1) = C_1 + C_2 e^{1/\varepsilon} + u_p(1) = 0$$
Using $k=11\pi$, we have $\sin(k)=0$ and $\cos(k)=-1$. Thus $u_p(1) = -B = \frac{1}{kD}$.
$$C_1 + C_2 e^{1/\varepsilon} = -\frac{1}{kD} \quad \text{(Eq. II)}$$

**Solving for $C_1, C_2$:**
Subtract (Eq. I) from (Eq. II):
$$
C_2 (e^{1/\varepsilon} - 1) = -\frac{2}{kD} \implies C_2 = -\frac{2}{kD(e^{1/\varepsilon} - 1)}
$$
From (Eq. I):
$$
C_1 = \frac{1}{kD} - C_2 = \frac{1}{kD} \left( 1 + \frac{2}{e^{1/\varepsilon} - 1} \right) = \frac{1}{kD} \left( \frac{e^{1/\varepsilon} + 1}{e^{1/\varepsilon} - 1} \right)
$$

## Final Closed-Form Solution
Combining the parts, with $k=11\pi$ and $D = 1 + (\varepsilon k)^2$:

$$
\boxed{
u(x) = \underbrace{\frac{1}{kD} \left[ \frac{e^{1/\varepsilon}+1 - 2e^{x/\varepsilon}}{e^{1/\varepsilon}-1} \right]}_{\text{Boundary Layer Term}} + \underbrace{\frac{\varepsilon \sin(kx) - \frac{1}{k}\cos(kx)}{D}}_{\text{Particular Solution}}
}
$$
