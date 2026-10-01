# Question Q2 — Localisation par EKF : le système complet

## 1. Objectif

Un robot se déplace dans un plan. Il doit estimer sa **pose** (position + orientation) à partir
de deux sources d'information :

1. **l'odométrie** : la mesure de son propre déplacement (source interne, qui dérive avec le temps) ;
2. **la perception d'amers** : des points de repère fixes et identifiables, dont on mesure la
   **distance** et la **direction** (source externe, qui permet de corriger la dérive).

Le filtre de Kalman étendu (EKF) **fusionne** ces deux sources pour produire une estimation
robuste de la pose au cours du temps.

---

## 2. État, commande et mesures

- **État** du robot (pose dans le repère global) :

$$
X_t = \begin{bmatrix} x_t & y_t & \theta_t \end{bmatrix}^\top
$$

- **Commande odométrique** (déplacement mesuré entre $t$ et $t+1$, exprimé dans le repère du robot) :

$$
U_t = \begin{bmatrix} x_u & y_u & \theta_u \end{bmatrix}^\top
$$

- **Mesure** d'un amer $k$ (distance + direction) :

$$
Y_t = \begin{bmatrix} r_t & \phi_t \end{bmatrix}^\top
$$

---

## 3. Qu'est-ce que l'odométrie ?

L'**odométrie** (ou *dead reckoning*) est la technique qui consiste à estimer sa position en
**intégrant** ses déplacements successifs. Le robot connaît (par ses encodeurs de roues, par
exemple) le déplacement qu'il pense avoir effectué à chaque pas :

$$
u_t = \begin{bmatrix} u_x & u_y & u_\theta \end{bmatrix}^\top
$$

Ce déplacement est exprimé dans le **repère local du robot** : $u_x$ (avant), $u_y$ (latéral),
$u_\theta$ (rotation).

### 3.1 La composition de poses (opérateur $\oplus$ de SE(2))

Pour passer de la pose $X$ à la nouvelle pose $X'$ en appliquant le déplacement $U$ :

$$
X' = X \oplus U
\qquad\Longleftrightarrow\qquad
\begin{cases}
\theta' = \theta + u_\theta \\[2pt]
\begin{bmatrix} x' \\ y' \end{bmatrix}
= \begin{bmatrix} x \\ y \end{bmatrix}
+ R(\theta)
\begin{bmatrix} u_x \\ u_y \end{bmatrix}
\end{cases}
$$

avec la matrice de rotation $R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$.
C'est la fonction `tcomp` du code.

### 3.2 Le bruit odométrique

L'odométrie est **bruitée** : le déplacement réellement mesuré est

$$
u_t = u_t^{\text{true}} + \eta_t,
\qquad
\eta_t \sim \mathcal{N}(0, Q_{\text{True}}),
\qquad
Q_{\text{True}} = \operatorname{diag}(0.01,\ 0.01,\ 1^\circ)^2 .
$$

En pratique, la pose odométrique $X^{\text{odom}}$ **intègre** ces déplacements bruités :

$$
X^{\text{odom}}_{t+1} = X^{\text{odom}}_t \oplus u_t .
$$

C'est pourquoi l'odométrie **dérive** : chaque petite erreur s'ajoute à la précédente, sans être
corrigée. Sur la figure, la trajectoire odométrique (verte) s'écarte peu à peu de la vraie
trajectoire (noire).

> **Rôle dans le filtre** : l'odométrie fournit la commande $u_t$ utilisée dans l'étape de
> **prédiction**. Le filtre ne connaît que la commande *bruitée*, pas la vraie.

---

## 4. Modèle d'évolution $f(X, U)$

La pose prédite à partir de la pose courante et du déplacement odométrique :

$$
X_{t+1} = f(X_t, U_t) =
\begin{bmatrix}
x_t + x_u \cos\theta_t - y_u \sin\theta_t \\[2pt]
y_t + x_u \sin\theta_t + y_u \cos\theta_t \\[2pt]
\theta_t + \theta_u
\end{bmatrix}
$$

C'est exactement la composition $X_t \oplus U_t$.

---

## 5. Les amers et le modèle d'observation $h(X)$

Les **amers** sont des points fixes du plan, de coordonnées connues $(x_k, y_k)$. Chaque amer
$k$ est supposé **parfaitement identifiable** (on sait quel amer on observe). La mesure
produite est le couple distance + direction :

$$
Y_t = h_k(X_t) =
\begin{bmatrix}
\sqrt{(x_k - x_t)^2 + (y_k - y_t)^2} \\[4pt]
\operatorname{atan2}(y_k - y_t,\ x_k - x_t) - \theta_t
\end{bmatrix}
=
\begin{bmatrix} r \\[2pt] \phi \end{bmatrix}
$$

avec un bruit gaussien de covariance $P_Y$ :

$$
P_{Y,\text{True}} = \operatorname{diag}(5.0,\ 6^\circ)^2 .
$$

> $r$ est la distance euclidienne robot–amer ; $\phi$ est l'angle sous lequel le robot voit
> l'amer, mesuré dans son propre repère (d'où le $- \theta_t$).

---

## 6. L'EKF : prédiction puis correction

### 6.1 Prédiction (à partir de l'odométrie)

$$
X_{\text{pred}} = f(X_{\text{est}}, u_t)
$$

$$
P_{\text{pred}} = A\, P_{\text{est}}\, A^\top + B\, Q_{\text{Est}}\, B^\top
$$

où $A = \partial f/\partial X$ et $B = \partial f/\partial U$ sont les jacobiennes du modèle
d'évolution (voir la section correspondante).

### 6.2 Correction (à partir de la perception d'un amer)

Innovation, covariance d'innovation et gain :

$$
\nu = Y_t - h_k(X_{\text{pred}}),
\qquad
S = H\, P_{\text{pred}}\, H^\top + P_{Y,\text{Est}},
\qquad
W = P_{\text{pred}}\, H^\top\, S^{-1}
$$

Mise à jour de la pose et de la covariance :

$$
X_{\text{est}} = X_{\text{pred}} + W\,\nu,
\qquad
P_{\text{est}} = P_{\text{pred}} - W\,H\,P_{\text{pred}}
$$

où $H = \partial h_k/\partial X$ est la jacobienne du modèle d'observation.

> L'angle $\theta$ et l'innovation angulaire sont **ramenés** dans $[-\pi, \pi]$ à chaque étape
> (fonction `angle_wrap`), car les angles sont définis modulo $2\pi$.

---

## 7. Bruits réels vs bruits supposés

Une subtilité importante du code : le bruit **réel** (utilisé par le simulateur) et le bruit
**supposé** (utilisé par le filtre) ne sont pas égaux.

| | Réel (simulateur) | Supposé (filtre) |
|---|---|---|
| Odométrie | $Q_{\text{True}}$ | $Q_{\text{Est}} = 10\, Q_{\text{True}}$ |
| Perception | $P_{Y,\text{True}}$ | $P_{Y,\text{Est}} = 10\, P_{Y,\text{True}}$ |

Le filtre est ici volontairement **pessimiste** : il suppose ses capteurs 10 fois plus bruités
qu'ils ne le sont réellement. Il accorde donc plus de poids à son modèle (prédiction) qu'à ses
mesures, ce qui le rend plus lisse mais plus prudent.

---

## 8. Résumé du pipeline (boucle du code)

À chaque pas $t$ :

1. **Simulation du vrai déplacement** : $X^{\text{true}}_{t+1} = X^{\text{true}}_t \oplus u^{\text{true}}_t$.
2. **Odométrie** : $u_t = u^{\text{true}}_t + \eta_t$, puis $X^{\text{odom}}_{t+1} = X^{\text{odom}}_t \oplus u_t$.
3. **Prédiction EKF** : $X_{\text{pred}} = f(X_{\text{est}}, u_t)$, $P_{\text{pred}} = A P A^\top + B Q_{\text{Est}} B^\top$.
4. **Perception** : choix d'un amer, mesure bruitée $Y_t = h_k(X^{\text{true}}_t) + \text{bruit}$.
5. **Correction EKF** : innovation, gain, mise à jour de $X_{\text{est}}$ et $P_{\text{est}}$.
6. **Sauvegarde** et affichage (trajectoires vraie / odométrique / estimée + ellipse de covariance).
