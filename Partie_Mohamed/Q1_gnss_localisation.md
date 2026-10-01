# Question Q1 — Localisation GNSS simplifiée

## 1. Contexte de la question

On applique la classe `KalmanFilter` à un problème de **localisation GNSS** (type GPS)
simplifié. L'objet se déplace dans $\mathbb{R}^3$ ; on ne mesure que sa **position**
(les mesures GNSS sont supposées indépendantes en choisissant $\Delta T$ suffisamment grand).

Sous-questions de l'énoncé :

- **(a)** Implémenter la simulation (prédiction / correction) avec le filtre de Kalman.
- **(b)** Simuler un **trou de mesure** entre $t=10$ et $t=20$ (exclus). Qu'observe-t-on ? Expliquer.
- **(c)** Faire varier le bruit de dynamique $Q$. Qu'observe-t-on ? Expliquer.
- **(d)** Faire varier le bruit de mesure $R$. Qu'observe-t-on ? Expliquer.

---

## 2. Modèle

### 2.1 État

L'état regroupe position et vitesse :

$$
x_k = \begin{bmatrix} p_x & p_y & p_z & v_x & v_y & v_z \end{bmatrix}^\top \in \mathbb{R}^6
$$

### 2.2 Modèle d'évolution

Pas de commande (dynamique à vitesse quasi-constante) :

$$
x_{k+1} = F\, x_k + w_k,
\qquad
w_k \sim \mathcal{N}(0, Q)
$$

avec la matrice de transition ($\Delta T = 1$) :

$$
F =
\begin{bmatrix}
1 & 0 & 0 & \Delta T & 0 & 0 \\
0 & 1 & 0 & 0 & \Delta T & 0 \\
0 & 0 & 1 & 0 & 0 & \Delta T \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
I_3 & \Delta T\,I_3 \\[2pt]
0 & I_3
\end{bmatrix}
$$

et la covariance du bruit de dynamique (valeurs de `tp1_part1.py`) :

$$
Q = \operatorname{diag}(10^{-4},\ 10^{-4},\ 10^{-4},\ 10^{-2},\ 10^{-2},\ 10^{-2})
$$

> Interprétation : le bruit sur la vitesse ($10^{-2}$) est nettement plus grand que celui
> sur la position ($10^{-4}$) : le modèle « laisse de la latitude » à la vitesse, ce qui
> lui permet de suivre un objet dont la vitesse réelle n'est pas exactement constante.

### 2.3 Modèle d'observation

On observe les trois coordonnées de position :

$$
y_k = H\, x_k + v_k,
\qquad
v_k \sim \mathcal{N}(0, R)
$$

$$
H =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix}
=
\begin{bmatrix}
I_3 & 0_{3\times 3}
\end{bmatrix},
\qquad
R = 100\,I_3
\quad (\sigma_{\text{mesure}} = 10\ \text{m})
$$

### 2.4 Initialisation

$$
x_0 = 0_6,
\qquad
P_0 = \operatorname{diag}(10^4,\ 10^4,\ 10^4,\ 10^2,\ 10^2,\ 10)
$$

État vrai initial : $x_0^{\text{true}} = [20,\,-10,\,5,\,0,\,5,\,0]^\top$.

---

## 3. Récapitulatif du filtre de Kalman

Étant donné $\hat{x}_{k|k}$ et $P_{k|k}$ :

**Prédiction**

$$
\hat{x}_{k+1|k} = F\,\hat{x}_{k|k}
\qquad
P_{k+1|k} = F\,P_{k|k}\,F^\top + Q
$$

**Correction**

$$
\begin{aligned}
S_{k+1} &= H\,P_{k+1|k}\,H^\top + R \\[2pt]
K_{k+1} &= P_{k+1|k}\,H^\top\,S_{k+1}^{-1} \\[2pt]
\hat{x}_{k+1|k+1} &= \hat{x}_{k+1|k} + K_{k+1}\,\big(y_{k+1} - H\,\hat{x}_{k+1|k}\big) \\[2pt]
P_{k+1|k+1} &= \big(I - K_{k+1} H\big)\,P_{k+1|k}
\end{aligned}
$$

(`kalman.py` utilise en réalité la forme de Joseph pour $P$, plus stable numériquement.)

---

## 4. Question (b) — Trou de mesure

Le masque `has_measure[11:21] = False` supprime la mesure pour $k \in [11, 20]$.
La boucle saute alors `simulate_y` et `update` : seules les étapes de **prédiction**
sont exécutées.

Pendant le trou :

$$
\hat{x}_{k+1|k+1} = \hat{x}_{k+1|k} = F\,\hat{x}_{k|k},
\qquad
P_{k+1|k+1} = P_{k+1|k} = F\,P_{k|k}\,F^\top + Q
$$

**Conséquences :**

1. **La covariance croît sans être recadrée.** Sans correction, $P$ ne peut que grandir
   (le terme $F P F^\top + Q$ est une somme de matrices définies positives), donc les
   bandes $\pm 3\sigma$ **s'élargissent** pendant tout le trou.

2. **L'estimation dérive.** Le filtre intègre uniquement le modèle à vitesse quasi-constante ;
   toute différence entre le modèle et la vraie trajectoire (bruit $w_k$) n'est plus corrigée,
   donc l'erreur d'estimation **augmente** avec le temps.

3. **Reconvergence à la reprise des mesures.** Dès que $y$ réapparaît, l'innovation
   $y - H\hat{x}$ est grande et la covariance élevée : le gain $K$ est important, le filtre
   **recale brutalement** l'estimation vers les mesures, puis $P$ **décroît** vers son régime
   stationnaire.

---

### 4.1 Preuve — croissance de la variance de position et de vitesse

On observe (sur les bandes $\pm 3\sigma$) que **la variance de position croît** alors que
**la variance de vitesse reste quasi constante**. On le prouve par calcul exact, avec la
matrice $F$ **complète** (6×6).

**Écriture avec la matrice complète.** L'état est $x = [p\ v]^\top$ avec
$p = [p_x\ p_y\ p_z]^\top$ (position) et $v = [v_x\ v_y\ v_z]^\top$ (vitesse).
Avec $\Delta T = 1$ :

$$
F =
\begin{bmatrix}
1 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
I_3 & I_3 \\[2pt]
0_3 & I_3
\end{bmatrix},
\qquad
Q =
\begin{bmatrix}
q_p I_3 & 0_3 \\[2pt]
0_3 & q_v I_3
\end{bmatrix},
\quad q_p = 10^{-4},\ q_v = 10^{-2}.
$$

**Covariance par blocs 3×3.** On partitionne la covariance complète :

$$
P_k =
\begin{bmatrix}
P_{pp,k} & P_{pv,k} \\[2pt]
P_{vp,k} & P_{vv,k}
\end{bmatrix}
$$

où $P_{pp,k}$ est la variance de position (3×3), $P_{vv,k}$ celle de vitesse, et
$P_{pv,k} = P_{vp,k}^\top$ la covariance croisée position–vitesse.

**Récurrence complète pendant le trou** (prédiction seule, $P_{k+1} = F P_k F^\top + Q$).
Le produit $F P_k F^\top$ donne :

$$
F P_k F^\top =
\begin{bmatrix}
P_{pp,k} + P_{pv,k} + P_{vp,k} + P_{vv,k} & P_{pv,k} + P_{vv,k} \\[2pt]
P_{vp,k} + P_{vv,k} & P_{vv,k}
\end{bmatrix}
$$

d'où les récurrences par blocs :

$$
\boxed{\begin{aligned}
P_{pp,k+1} &= P_{pp,k} + P_{pv,k} + P_{vp,k} + P_{vv,k} + q_p\,I_3 \\[2pt]
P_{pv,k+1} &= P_{pv,k} + P_{vv,k} \\[2pt]
P_{vp,k+1} &= P_{vp,k} + P_{vv,k} \\[2pt]
P_{vv,k+1} &= P_{vv,k} + q_v\,I_3
\end{aligned}}
$$

**Découplage par axe.** $F$, $Q$, $H$, $R$ étant diagonales par blocs (un bloc 2×2 par axe),
tous les blocs $P_{pp}$, $P_{pv}$, $P_{vp}$, $P_{vv}$ sont eux-mêmes **diagonaux**. Chaque
axe $(x,y,z)$ suit donc une récurrence scalaire indépendante. Pour un axe donné, on pose

$$
a_k = (P_{pp,k})_{ii},\quad b_k = (P_{pv,k})_{ii} = (P_{vp,k})_{ii},\quad c_k = (P_{vv,k})_{ii},
$$

et les récurrences par blocs se réduisent exactement à :

$$
\boxed{\begin{aligned}
a_{k+1} &= a_k + 2b_k + c_k + q_p \\[2pt]
b_{k+1} &= b_k + c_k \\[2pt]
c_{k+1} &= c_k + q_v
\end{aligned}}
$$

**1) Variance de vitesse : croissance linéaire (pente faible).**

La troisième récurrence donne immédiatement

$$
c_k = c_0 + k\,q_v .
$$

La variance de vitesse croît donc **linéairement** en $k$, avec une pente $q_v = 10^{-2}$
très faible. Sur 10 pas : $c_{10} - c_0 = 0.1$, à comparer à $c_0 \approx 0.14$ (régime
stationnaire). L'écart-type

$$
\sigma_v(k) = \sqrt{c_0 + k\,q_v}
$$

varie peu : c'est pour cela que la bande $\pm 3\sigma_v$ **semble constante**.

**2) Variance de position : croissance quadratique (pente croissante).**

De $b_{k+1} = b_k + c_k$ on déduit

$$
b_k = b_0 + k\,c_0 + q_v\,\frac{k(k-1)}{2} .
$$

En injectant dans la récurrence de $a$ puis en sommant, on obtient la forme fermée :

$$
\boxed{a_k = a_0 + (2b_0 + c_0 + q_p)\,k + c_0\,k(k-1) + q_v\,\frac{k(k-1)(2k-1)}{6}}
$$

Le terme dominant est **quadratique** : $a_k \sim c_0\,k^2$ (le terme cubique en $q_v$ est
négligeable devant lui). En effet la dérivée seconde discrète vaut

$$
\Delta^2 a_k := \Delta a_{k+1} - \Delta a_k = 2c_k + q_v = 2c_0 + q_v + 2 q_v k \approx 2c_0,
$$

constante au premier ordre : c'est la signature d'une croissance **quadratique**.

L'écart-type, lui, croît **linéairement** :

$$
\sigma_p(k) = \sqrt{a_k} \sim \sqrt{c_0}\ k .
$$

**Conclusion.** C'est bien l'écart-type $\sigma$ (ce qui est tracé) qui est linéaire pour la
position et quasi constant pour la vitesse. En termes de variance :
- position : **quadratique** en $k$ (terme dominant $c_0 k^2$) ;
- vitesse : **linéaire** en $k$, mais de pente $q_v$ si faible qu'elle paraît constante.

> L'intuition physique : la vitesse se propage par simple ajout de bruit ($+q_v$ à chaque pas),
> donc sa variance augmente d'une constante par pas. La position, elle, intègre la vitesse
> ($p_{k+1} = p_k + v_k$), donc son incertitude **s'accumule** : chaque pas ajoute l'incertitude
> de vitesse accumulée, d'où la croissance accélérée (quadratique) de sa variance.

---

### 4.2 Interprétation — pourquoi, vraiment, sans les maths

**Le filtre privé de mesure « navigue à l'aveugle ».** Tant qu'il voit le GPS, chaque
mesure lui sert de point d'ancrage : elle le « recale » et écrase en grande partie
l'erreur accumulée. Dès que la mesure disparaît, il ne lui reste que son modèle de
mouvement, et il doit **extrapoler** dans le futur. Rien ne vient plus corriger le tir :
les erreurs ne font que s'additionner d'un pas à l'autre, sans jamais être remises à zéro.

**Pourquoi la vitesse reste (presque) stable.** Le modèle dit « la vitesse ne change pas,
à peu de chose près ». À chaque pas, le filtre ajoute donc un petit doute fixe sur la
vitesse : un petit coup de bruit, toujours de même taille. Ce doute ne se **répercute pas
sur lui-même** — la vitesse de demain ne dépend pas de la position, seulement de la vitesse
d'aujourd'hui plus un petit aléa constant. L'incertitude s'accumule donc lentement et
régulièrement, comme des grains de sable qui tombent un par un : ça monte, mais de façon
paisible, quasi invisible sur la durée du trou.

**Pourquoi la position s'emballe.** La position, elle, est obtenue en **intégrant la
vitesse** : chaque pas, on ajoute la vitesse estimée pour avancer. Or cette vitesse est
elle-même entachée d'incertitude. Un petit doute sur la vitesse se transforme donc, pas
après pas, en une erreur de position **qui grandit**. Et comme l'incertitude de vitesse
s'accumule aussi, l'erreur de position s'ajoute à une erreur de position déjà gonflée :
c'est un effet « boule de neige ». Plus le temps passe, plus on est loin, et plus vite on
s'enfonce.

**L'analogie du marcheur dans le brouillard.** Imaginez quelqu'un qui avance à l'aveugle
en pensant aller tout droit mais avec un cap légèrement faux :
- à chaque pas, son **angle** d'erreur reste grosso modo le même (comme la vitesse) ;
- mais la **distance** qui le sépare du bon chemin grandit de plus en plus vite, car chaque
  pas l'éloigne un peu plus dans la mauvaise direction (comme la position).

Ou, plus simple : c'est l'écart entre « je crois rouler à 30 km/h » et « je roule vraiment
à 30 km/h » qui ne bouge presque pas, alors que la **distance parcourue** estimée dérive de
plus en plus vite, car l'erreur de compteur est intégrée à chaque kilomètre.

**En résumé.** La vitesse ne subit qu'une **dérive additive constante** (doute stable),
donc son incertitude n'explose pas. La position **accumule l'erreur de vitesse**, donc son
incertitude se nourrit d'elle-même et croît de façon accélérée : c'est le comportement
typique de la **navigation à l'estime** (dead reckoning) sans recalage externe.

---

## 5. Question (c) — Variation de $Q$ (bruit de dynamique)

$Q$ quantifie la confiance du filtre dans son **modèle de mouvement**.

- **$Q \uparrow$** : $P_{k+1|k} = F P F^\top + Q$ plus grand $\Rightarrow$ $K$ plus grand
  $\Rightarrow$ le filtre accorde **plus de poids aux mesures**. Résultat : estimation
  **réactive**, qui suit rapidement les changements, mais **plus bruitée** (les bruits de
  mesure sont davantage recopiés dans l'estimation).

- **$Q \downarrow$** : le filtre est **confiant dans son modèle** $\Rightarrow$ $K$ plus
  petit $\Rightarrow$ estimation **plus lisse**, mais **plus lente** à réagir : elle suit
  moins bien la vraie dynamique et se recale plus lentement après le trou de mesure.

En limite, $Q \to 0$ avec un modèle parfait : le filtre « ferme les yeux » sur les mesures ;
à l'inverse, $Q \to \infty$ : le filtre se réduit à un estimateur qui recopie quasiment les
observations.

---

## 6. Question (d) — Variation de $R$ (bruit de mesure)

$R$ quantifie la confiance du filtre dans ses **capteurs**.

- **$R \uparrow$** : $S = H P H^\top + R$ plus grand $\Rightarrow$ $K$ plus petit
  $\Rightarrow$ les mesures sont jugées **peu fiables**, le filtre les pondère faiblement :
  estimation **lissée** mais moins précise (l'erreur reste plus élevée).

- **$R \downarrow$** : les mesures sont jugées **quasi parfaites** $\Rightarrow$ $K$ plus
  grand $\Rightarrow$ l'estimation **colle** aux observations, mais **amplifie** le bruit de
  mesure dans l'estimation.

Cas limite $R \to 0$ : le filtre force $H \hat{x} \approx y$ (collage aux mesures) ;
$R \to \infty$ : les mesures sont ignorées, le filtre se réduit à la prédiction pure.

---

## 7. Régime stationnaire (équation de Riccati)

Pour un modèle linéaire constant $(F, H, Q, R)$ avec correction à chaque pas, la covariance
d'erreur converge vers un point fixe $P_{\infty}$ solution de l'équation de **Riccati discrète** :

$$
P_{\infty} = F\,P_{\infty}\,F^\top + Q
- F\,P_{\infty}\,H^\top\,\big(H\,P_{\infty}\,H^\top + R\big)^{-1} H\,P_{\infty}\,F^\top
$$

et le gain limite vaut

$$
K_{\infty} = P_{\infty}\,H^\top\,\big(H\,P_{\infty}\,H^\top + R\big)^{-1}.
$$

C'est ce régime stationnaire qu'on observe après la phase transitoire initiale : $P$ décroît
depuis $P_0$ puis se stabilise, et les bandes $\pm 3\sigma$ gardent une largeur constante.
Pendant le trou de mesure, on sort de ce régime (prédiction seule $\Rightarrow$ $P$ croît),
puis on y **reconverge** dès que les mesures reprennent.

---

## 8. Correspondance avec `tp1_part1.py`

| Notion | Symbole | Dans le code |
|---|---|---|
| État | $x$ | `XTrue` / `kf.x` |
| Covariance | $P$ | `PEst` / `kf.P` |
| Transition | $F$ | `Fk` |
| Bruit de dynamique | $Q$ | `Qk` |
| Observation | $H$ | `Hk` |
| Bruit de mesure | $R$ | `Rk` |
| Trou de mesure | — | `has_measure[11:21] = False` |
