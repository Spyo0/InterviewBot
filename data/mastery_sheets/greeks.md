# Fiche de Maîtrise : Les Grecques (Greeks)

## Théorie

Les Grecques mesurent la sensibilité du prix d'une option à chaque paramètre de marché. Elles sont centrales pour le risk management et la couverture.

**Delta** $\Delta$ — Sensibilité au prix du sous-jacent :
$$\Delta_{\text{call}} = \Phi(d_1), \quad \Delta_{\text{put}} = \Phi(d_1) - 1 = -\Phi(-d_1)$$

**Gamma** $\Gamma$ — Convexité (dérivée seconde en $S$, identique pour call et put) :
$$\Gamma = \frac{\phi(d_1)}{S\sigma\sqrt{T}}$$

où $\phi$ est la PDF normale standard. Gamma est maximal ATM et diminue ITM/OTM.

**Vega** $\mathcal{V}$ — Sensibilité à la volatilité implicite :
$$\mathcal{V} = S\phi(d_1)\sqrt{T}$$

Même signe pour call et put (toujours positif pour un acheteur d'option).

**Theta** $\Theta$ — Décroissance temporelle (time decay) :
$$\Theta_{\text{call}} = -\frac{S\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}\Phi(d_2)$$

Theta est généralement négatif pour un acheteur d'option : le temps détruit de la valeur.

**Rho** $\rho$ — Sensibilité au taux sans risque :
$$\rho_{\text{call}} = KTe^{-rT}\Phi(d_2)$$

**Relation BS fondamentale (theta-gamma-vega)** :
$$\Theta + \frac{1}{2}\sigma^2 S^2 \Gamma + rS\Delta - rV = 0$$

Pour un portefeuille delta-neutre : $\Theta \approx -\frac{1}{2}\sigma^2 S^2 \Gamma$. **Gamma et Theta sont en opposition constante.**

---

## Insights Forums

- **"Expliquez le trade-off Gamma/Theta."** — Être long Gamma (long option) coûte du Theta chaque jour. Le P&L d'un delta-hedger = $\frac{1}{2}\Gamma S^2[(\sigma_{\text{réalisée}})^2 - (\sigma_{\text{implicite}})^2]dt$. Si $\sigma_{\text{réalisée}} > \sigma_{\text{implicite}}$, on gagne.
- **"Quel Greek est le plus important pour un trader de vol ?"** — Vega et Gamma. Vega mesure l'exposition statique à la vol, Gamma mesure le bénéfice dynamique de la convexité.
- **"Delta d'un portefeuille en pratique."** — En gestion de book, on calcule le dollar delta ($S \cdot \Delta$) et on ajuste la position en sous-jacent pour rester delta-neutre.
- **"Comment varie Gamma avec $T$ ?"** — Gamma explose à l'approche de l'expiration pour les options ATM. C'est le "pin risk" redouté des market makers.

---

## Analyse des Limites

| Situation | Limite des Grecques classiques |
|---|---|
| **Volatilité stochastique** | Vega n'est plus une mesure unique : il faut "vega par strike" (volga, vanna) pour gérer le smile. |
| **Discontinuités de payoff** | Gamma infini au strike pour les options digitales (Dirac) : impossible à couvrir parfaitement. |
| **Couverture discrète** | Les Grecques supposent un rebalancement continu. En pratique, le slippage de couverture crée un risque résiduel. |
| **Greeks de second ordre** | Volga ($\partial \mathcal{V}/\partial\sigma$) et Vanna ($\partial \Delta/\partial\sigma$) sont nécessaires pour gérer le smile. Ignorés par BS simple. |
| **Options américaines** | Delta peut "sauter" à la frontière d'exercice optimal. Pas de formule fermée pour les Grecques. |

---

## Contre-intuitions (Pièges recruteurs)

1. **"Le Theta est toujours négatif."** — Faux. Un put européen très ITM sur actif avec taux élevé peut avoir un Theta positif (la valeur de l'option croît avec le temps car on veut exercer mais on ne peut pas).

2. **"Gamma d'un call et d'un put sont différents."** — Faux. Par put-call parity, $\Gamma_{\text{call}} = \Gamma_{\text{put}}$ pour le même strike et maturité. La formule est identique.

3. **"Vega augmente avec la maturité."** — Vrai ($\mathcal{V} \propto \sqrt{T}$), mais la sensibilité *relative* (vega/prix) peut varier. Un deep-OTM court terme peut avoir un fort vega relatif.

4. **"Delta d'un call ATM vaut 0.5."** — Approximativement vrai, mais seulement quand $r = 0$ et $T$ petit. Précisément, $\Delta_{\text{ATM}} = \Phi(d_1)$ avec $d_1 = \frac{(r + \sigma^2/2)T}{\sigma\sqrt{T}} \neq 0$ en général.

5. **"Pour couvrir une option, il suffit de delta-hedger."** — Le delta-hedge élimine le risque de premier ordre. Mais le Gamma (risque de convexité) reste exposé. Un gamma-hedge nécessite une autre option dans le portefeuille.
