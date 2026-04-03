# Fiche de Maîtrise : Méthodes de Monte Carlo en Finance

## Théorie

Les méthodes Monte Carlo estiment une espérance $\mathbb{E}^{\mathbb{Q}}[e^{-rT} f(S_T)]$ par simulation numérique. Elles sont particulièrement adaptées aux payoffs path-dependent et multi-actifs.

**Algorithme de base pour un call européen sur GBM** :

1. Simuler $N$ trajectoires : $S_T^{(i)} = S_0 \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\, Z_i\right]$, $Z_i \sim \mathcal{N}(0,1)$
2. Calculer le payoff : $f_i = \max(S_T^{(i)} - K, 0)$
3. Estimer le prix : $\hat{V} = e^{-rT} \frac{1}{N}\sum_{i=1}^N f_i$

**Erreur standard** :
$$\text{SE} = \frac{\hat{\sigma}_f}{\sqrt{N}}$$

L'erreur décroît en $O(1/\sqrt{N})$. Pour réduire l'erreur de moitié : multiplier $N$ par 4.

**Techniques de réduction de variance** :

- **Variables antithétiques** : pour chaque $Z_i$, simuler aussi $-Z_i$. Divise la variance si $\text{Cov}(f(Z), f(-Z)) < 0$.
- **Variable de contrôle** : si on connaît $\mathbb{E}[g]$ analytiquement, utiliser $\hat{V}_{\text{cv}} = \hat{V} - \beta(\hat{g} - \mathbb{E}[g])$ avec $\beta = \text{Cov}(f,g)/\text{Var}(g)$.
- **Quasi-Monte Carlo (QMC)** : remplacer les nombres pseudo-aléatoires par des suites à faible discrépance (Sobol, Halton). Convergence en $O((\ln N)^d / N)$ vs $O(1/\sqrt{N})$.
- **Importance Sampling** : changer de mesure pour sur-représenter les scénarios importants (OTM extrêmes).

**Pricing d'options path-dependent** (e.g., option asiatique) :
$$\hat{V} = e^{-rT} \frac{1}{N}\sum_{i=1}^N \max\!\left(\frac{1}{M}\sum_{j=1}^M S_{t_j}^{(i)} - K,\ 0\right)$$

---

## Insights Forums

- **"Combien de simulations pour une précision de 1 bp ?"** — SE $\approx \sigma_f / \sqrt{N}$. Pour $\sigma_f \approx 5$ et SE $= 0.01$ : $N \approx 250\,000$ simulations. Avec antithétiques : $N \approx 62\,500$.
- **"Comment pricer une option barrière en MC ?"** — Simuler des trajectoires discrètes et vérifier si la barrière est franchie à chaque pas. Biais de discrétisation car la trajectoire continue peut franchir la barrière entre deux pas (correction de Broadie-Glasserman : ajuster avec $e^{-2h(h-b)/(sigma^2 dt)}$).
- **"Calculez le delta par MC."** — Méthode bump-and-reprice : $\Delta \approx \frac{\hat{V}(S_0 + h) - \hat{V}(S_0 - h)}{2h}$. Ou par pathwise differentiation / likelihood ratio method (plus efficace mais plus complexe).
- **"Qu'est-ce que la méthode de Longstaff-Schwartz ?"** — Algorithme pour pricer les options américaines par MC : régressions backward pour estimer la valeur de continuation à chaque pas de temps.

---

## Analyse des Limites

| Situation | Problème |
|---|---|
| **Convergence lente** | L'erreur en $O(1/\sqrt{N})$ rend MC coûteux pour les hautes précisions. Les méthodes EDP (différences finies) convergent plus vite en basse dimension. |
| **Haute dimension** | La malédiction de la dimensionnalité frappe les méthodes EDP (grille exponentielle). MC garde $O(1/\sqrt{N})$ indépendamment de $d$ : avantage pour $d > 3$. |
| **Options américaines** | L'exercice anticipé crée un problème d'optimisation dynamique. MC simple est inadapté ; Longstaff-Schwartz ou tree methods nécessaires. |
| **Queue de distribution** | Rare events (stress scenarios) sont sous-représentés. Importance sampling ou techniques d'amplification de variance nécessaires. |
| **Calibration** | Calibrer un modèle par MC est très coûteux (millions de simulations par itération). Les formules analytiques ou semi-analytiques sont préférées pour la calibration. |

---

## Contre-intuitions (Pièges recruteurs)

1. **"Plus on augmente $N$, plus l'erreur diminue linéairement."** — Faux. L'erreur diminue en $\sqrt{N}$ : doubler $N$ ne divise l'erreur que par $\sqrt{2} \approx 1.41$.

2. **"Les variables antithétiques divisent toujours la variance par 2."** — Non. La réduction dépend de $\text{Cov}(f(Z), f(-Z))$. Pour un call, la réduction est significative car $f(Z)$ et $f(-Z)$ sont anti-corrélés. Pour un straddle, la corrélation peut être positive et les antithétiques inutiles.

3. **"MC est toujours plus lent que les méthodes analytiques."** — Vrai pour les options vanilles en BS, mais MC est souvent la *seule* méthode praticable pour les payoffs exotiques, multi-sous-jacents ou sous modèles complexes (Heston + sauts).

4. **"Un générateur de nombres aléatoires standard suffit."** — Le Mersenne Twister est standard, mais ses corrélations de long terme peuvent biaiser les simulations de haute dimension. Les suites de Sobol (QMC) offrent une meilleure couverture de l'espace.

5. **"Le pricing MC en monde risk-neutral utilise $\mu$."** — Non. On simule sous $\mathbb{Q}$ avec $\mu$ remplacé par $r$. Le drift physique $\mu$ est hors-sujet pour le pricing (mais pas pour la simulation de scénarios risk management).
