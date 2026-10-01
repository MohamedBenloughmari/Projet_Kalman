# Question Q0 — Filtre de Kalman (a) et filtre de Kalman étendu (b)

## Q0a — Filtre de Kalman linéaire (position + vitesse 1D)

### Contexte

Construire la classe `KalmanFilter` (méthodes `predict` et `update`), puis la tester sur
l'exemple du cours : un objet qui se déplace sur un axe, avec état position + vitesse.

### Modèle

État : $x = [p\ v]^\top$ (position, vitesse), $\Delta T = 1$.

**Évolution**

$$
x_{k+1} = F\,x_k + w_k,
\qquad
F = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix},
\qquad
w_k \sim \mathcal{N}\!\left(0,\ Q\right),
\quad
Q = \begin{bmatrix} 1 & 0 \\ 0 & 0.001 \end{bmatrix}
$$

**Observation** (on mesure la position seule)

$$
y_k = H\,x_k + v_k,
\qquad
H = \begin{bmatrix} 1 & 0 \end{bmatrix},
\qquad
v_k \sim \mathcal{N}(0,\ R),\quad R = [1]
$$

**Initialisation** : réel $x_0 = [20,\ 12]^\top$, estimation $\hat{x}_0 = [0,\ 10]^\top$,
covariance $P_0 = \operatorname{diag}(100,\ 1)$.

### Récapitulatif des équations (rappel)

$$
\begin{aligned}
\text{Prédiction :}&& \hat{x}_{k|k-1} &= F\,\hat{x}_{k-1|k-1}, & P_{k|k-1} &= F P_{k-1|k-1} F^\top + Q \\[2pt]
\text{Innovation :}&& \nu_k &= y_k - H\,\hat{x}_{k|k-1}, & S_k &= H P_{k|k-1} H^\top + R \\[2pt]
\text{Gain :}&& K_k &= P_{k|k-1} H^\top S_k^{-1} \\[2pt]
\text{Correction :}&& \hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_k\,\nu_k, & P_{k|k} &= (I - K_k H)\,P_{k|k-1}
\end{aligned}
$$

### Vérification numérique au pas $k=1$

**Prédiction**

$$
\hat{x}_{1|0} = F \hat{x}_0 =
\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} 0 \\ 10 \end{bmatrix}
= \begin{bmatrix} 10 \\ 10 \end{bmatrix}
$$

$$
P_{1|0} = F P_0 F^\top + Q =
\begin{bmatrix} 100 & 1 \\ 1 & 1 \end{bmatrix} +
\begin{bmatrix} 1 & 0 \\ 0 & 0.001 \end{bmatrix}
=
\begin{bmatrix} 102 & 1 \\ 1 & 1.001 \end{bmatrix}
$$

**Correction** (mesure $y_1 = 29.91$)

$$
S_1 = H P_{1|0} H^\top + R = 102 + 1 = 103,
\qquad
K_1 = P_{1|0} H^\top S_1^{-1} = \frac{1}{103}\begin{bmatrix} 102 \\ 1 \end{bmatrix}
= \begin{bmatrix} 0.9903 \\ 0.0097 \end{bmatrix}
$$

$$
\nu_1 = 29.91 - 10 = 19.91
\quad\Longrightarrow\quad
\hat{x}_{1|1} = \begin{bmatrix} 10 \\ 10 \end{bmatrix} + K_1 \cdot 19.91
= \begin{bmatrix} 29.72 \\ 10.19 \end{bmatrix}
$$

$$
P_{1|1} = (I - K_1 H) P_{1|0} =
\begin{bmatrix} 0.9903 & 0.0097 \\ 0.0097 & 0.9913 \end{bmatrix}
$$

> On retrouve exactement les valeurs du cours ($\hat{x}_{1|1} \approx [29.7,\ 10.2]$,
> $P_{1|1}$ donné).

### Comportement de la variance de vitesse

La position est **directement mesurée**, et la vitesse est **observable** par couplage
($p_{k+1} = p_k + v_k$) : au fil des mesures, la vitesse se déduit des positions. La
variance de vitesse **décroît** donc et **converge** vers une valeur stationnaire
(ici $\operatorname{var}(v)$ passe de $1$ à $\approx 0.12$ en 10 pas, en décroissance
monotone). Rien ne la fait croître : la correction « tire » l'incertitude vers le bas à
chaque pas.

---

## Q0b — Filtre de Kalman étendu (mesure d'angle uniquement)

### Contexte

Un objet se déplace sur l'axe $x$ ; son état est $[s\ v]^\top$ (position, vitesse). Un
capteur placé en $(x_s, y_s) = (0, -20)$ mesure un **angle bruité** vers l'objet (pas de
mesure de distance). On construit la classe `ExtendedKalmanFilter` et on la teste sur cet
exemple.

### Modèle

**Évolution** (linéaire ici, $\Delta T = 1$)

$$
x_{k+1} = F x_k + w_k,
\qquad
F = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix},
\qquad
w_k \sim \mathcal{N}(0, Q),
\quad
Q = \begin{bmatrix} 0.0001 & 0 \\ 0 & 0.01 \end{bmatrix}
$$

**Observation** (non linéaire — mesure d'angle)

$$
z_k = h(s_k) + v_k,
\qquad
h(s) = \operatorname{atan2}\!\big(-y_s,\ s - x_s\big) = \operatorname{atan2}(20,\ s),
\qquad
v_k \sim \mathcal{N}(0, R)
$$

avec $R = (1^\circ\ \text{en radians})^2 \approx (0.01745)^2 \approx 3.05\times 10^{-4}$.

**Jacobienne d'observation** (l'EKF linéarise $h$)

$$
H = \frac{\partial h}{\partial x} =
\begin{bmatrix} \dfrac{\partial h}{\partial s} & \dfrac{\partial h}{\partial v} \end{bmatrix}
=
\begin{bmatrix} h_s & 0 \end{bmatrix},
\qquad
h_s = \frac{y_s}{(s - x_s)^2 + y_s^2} = \frac{-20}{s^2 + 400}
$$

> Point clé : $h_s$ **dépend de $s$** et tend vers $0$ quand l'objet s'éloigne du capteur
> ($s \to \infty$). La mesure devient alors de moins en moins informative sur la position.

**Initialisation** : réel $x_0 = [2,\ 4]^\top$, estimation $\hat{x}_0 = [0,\ 5]^\top$,
$P_0 = \operatorname{diag}(100,\ 1)$.

### Récapitulatif de l'EKF

$$
\begin{aligned}
\text{Prédiction :}&& \hat{x}_{k|k-1} &= f(\hat{x}_{k-1|k-1}), & P_{k|k-1} &= F P_{k-1|k-1} F^\top + Q \\[2pt]
\text{Innovation :}&& \nu_k &= z_k - h(\hat{x}_{k|k-1}), & S_k &= H P_{k|k-1} H^\top + R \\[2pt]
\text{Gain :}&& K_k &= P_{k|k-1} H^\top S_k^{-1} \\[2pt]
\text{Correction :}&& \hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_k \nu_k, & P_{k|k} &= (I - K_k H) P_{k|k-1}
\end{aligned}
$$

### Observation : « converge puis diverge »

Au début, les mesures d'angle réduisent fortement l'incertitude (la position est « vue »
avec un angle sensible). Puis, à mesure que l'objet s'éloigne du capteur, $h_s \to 0$ : la
mesure devient quasi muette, et les covariances **se remettent à croître** sans limite —
c'est la divergence observée.

---

### Preuve — pourquoi la variance de vitesse augmente

On note la covariance $P = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$, avec
$a = \operatorname{var}(s)$, $c = \operatorname{var}(v)$ (variance de vitesse) et
$b = \operatorname{cov}(s,v)$.

**Prédiction** ($F P F^\top + Q$, avec $q_p = 10^{-4}$, $q_v = 10^{-2}$) :

$$
a_{\text{pred}} = a + 2b + c + q_p,
\qquad
b_{\text{pred}} = b + c,
\qquad
c_{\text{pred}} = c + q_v .
$$

**Correction** avec $H = [h_s\ 0]$ :

$$
S = h_s^2\, a_{\text{pred}} + R,
\qquad
K = P_{\text{pred}} H^\top S^{-1}
= \frac{1}{S}\begin{bmatrix} h_s\,a_{\text{pred}} \\ h_s\,b_{\text{pred}} \end{bmatrix}.
$$

Le bloc $(2,2)$ de $P_{\text{corr}} = (I - KH) P_{\text{pred}}$ donne la nouvelle variance de
vitesse :

$$
c_{\text{corr}} = c_{\text{pred}} - \frac{h_s^2\, b_{\text{pred}}^2}{S}
= c_{\text{pred}} - \frac{h_s^2\, b_{\text{pred}}^2}{h_s^2\, a_{\text{pred}} + R}.
$$

D'où la récurrence complète :

$$
\boxed{\,c_{k+1} = c_k + q_v - \frac{h_s^2\,(b_k + c_k)^2}{\,h_s^2\,(a_k + 2b_k + c_k + q_p) + R\,}\,}
$$

**Lecture de la formule**

1. **Terme $+q_v$** : à chaque prédiction, le bruit de dynamique injecte $q_v = 10^{-2}$
   d'incertitude sur la vitesse. C'est le moteur de la croissance.

2. **Terme de correction $-\dfrac{h_s^2 b_{\text{pred}}^2}{S}$** : c'est la seule chose qui
   « tire » la variance de vitesse vers le bas. Or la mesure d'angle n'observe **pas**
   directement la vitesse ($H$ a une composante nulle sur $v$) : la vitesse n'est corrigée
   qu'**indirectement**, via sa covariance croisée $b$ avec la position.

3. **Rôle de $h_s \to 0$** : quand l'objet s'éloigne, $h_s = -20/(s^2+400) \to 0$. Alors
   le numérateur $h_s^2 b_{\text{pred}}^2 \to 0$ (et $S \to R$, non nul), donc le terme de
   correction **s'annule**. La récurrence se réduit à

   $$
   c_{k+1} \approx c_k + q_v .
   $$

   La variance de vitesse croît alors **linéairement**, avec une pente $q_v = 10^{-2}$.

**Vérification numérique** (pas 8 → 9) : $c = 0.0491$, $b = 0.1128$, $h_s = -0.01085$,
d'où $c_{\text{pred}} = 0.0591$ et terme de correction $\approx 0.0078$, soit
$c_{\text{new}} \approx 0.0513$ — la simulation donne $0.0514$. ✓

> La même mécanique (et plus violente encore) s'applique à la position : $a$ croît aussi,
> car $a_{\text{pred}} = a + 2b + c + q_p$ intègre $c$ qui croît. C'est pourquoi la courbe
> de position diverge le plus vite.

### Interprétation intuitive (sans les maths)

Le capteur ne mesure qu'un **angle**. À courte distance, cet angle « bouge » beaucoup quand
l'objet bouge : il est très informatif, et le filtre estime bien position **et** vitesse.
Mais à mesure que l'objet **s'éloigne**, un même déplacement produit une variation d'angle
de plus en plus minuscule — le capteur devient pratiquement aveugle.

À ce moment, la mesure ne peut plus freiner la dérive du modèle : à chaque pas le filtre
ajoute son doute habituel sur la vitesse ($q_v$), et plus rien ne le corrige. La vitesse,
qu'on ne voit jamais directement, n'est recadrée qu'**à travers** la position ; dès que ce
lien se détend (angle insensible), son incertitude repart à la hausse, pas après pas, sans
limite.

Analogie : c'est comme estimer la vitesse d'un objet lointain en observant son
**déplacement angulaire**. Quand il est près de vous, vous voyez bien bouger votre cible ;
quand il file à l'horizon, vous ne savez plus s'il avance vite ou lentement — et votre
incertitude sur sa vitesse ne cesse de grandir.

### Résumé des deux questions

| | Q0a (Kalman) | Q0b (EKF) |
|---|---|---|
| Mesure | position (linéaire) | angle (non linéaire) |
| Vitesse observable ? | oui (via position) | seulement indirectement, et de moins en moins |
| Variance de vitesse | décroît puis **converge** | décroît puis **augmente (diverge)** |
| Cause | mesure directe et informative | $h_s \to 0$ : la mesure perd son information, $c_{k+1} \approx c_k + q_v$ |
