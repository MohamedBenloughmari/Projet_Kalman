# Q2-1 — Calcul des jacobiennes (EKF Localization)

Notations : état $X = [x\ y\ \theta]^\top$, commande $U = [x_u\ y_u\ \theta_u]^\top$,
mesure $Y = [r\ \phi]^\top$. Trois jacobiennes : $A = \partial f/\partial X$,
$B = \partial f/\partial U$, $H = \partial h/\partial X$.

## 1. Modèle d'évolution

$$
x' = x + x_u\cos\theta - y_u\sin\theta,\quad
y' = y + x_u\sin\theta + y_u\cos\theta,\quad
\theta' = \theta + \theta_u
$$

**Jacobienne $A = \partial f/\partial X$**

$$
A =
\begin{bmatrix}
1 & 0 & -x_u\sin\theta - y_u\cos\theta \\[2pt]
0 & 1 & \phantom{-}x_u\cos\theta - y_u\sin\theta \\[2pt]
0 & 0 & 1
\end{bmatrix}
$$

**Jacobienne $B = \partial f/\partial U$** (rotation $R(\theta)$ + $1$ sur l'angle)

$$
B =
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\[2pt]
\sin\theta & \phantom{-}\cos\theta & 0 \\[2pt]
0 & 0 & 1
\end{bmatrix}
$$

## 2. Modèle d'observation

Avec $dx = x_k - x$, $dy = y_k - y$, $r = \sqrt{dx^2+dy^2}$ :

$$
h(X) =
\begin{bmatrix} r \\[2pt] \operatorname{atan2}(dy,\ dx) - \theta \end{bmatrix}
$$

**Jacobienne $H = \partial h/\partial X$**

Distance : $\dfrac{\partial r}{\partial x} = -\dfrac{dx}{r}$, $\dfrac{\partial r}{\partial y} = -\dfrac{dy}{r}$, $\dfrac{\partial r}{\partial\theta} = 0$.

Direction (rappel $\partial\operatorname{atan2}(y,x)/\partial x = -y/r^2$, $\partial\operatorname{atan2}(y,x)/\partial y = x/r^2$) :
$\dfrac{\partial\phi}{\partial x} = \dfrac{dy}{r^2}$, $\dfrac{\partial\phi}{\partial y} = -\dfrac{dx}{r^2}$, $\dfrac{\partial\phi}{\partial\theta} = -1$.

$$
H =
\begin{bmatrix}
-\dfrac{dx}{r} & -\dfrac{dy}{r} & 0 \\[8pt]
\phantom{-}\dfrac{dy}{r^2} & -\dfrac{dx}{r^2} & -1
\end{bmatrix}
$$

## 3. Usage

- Prédiction : $P_{\text{pred}} = A P A^\top + B\,Q_{\text{Est}}\,B^\top$.
- Correction : $S = H P_{\text{pred}} H^\top + P_Y$, $W = P_{\text{pred}} H^\top S^{-1}$.
