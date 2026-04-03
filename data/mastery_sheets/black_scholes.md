# Fiche de Maîtrise : Modèle Black-Scholes

## Théorie

**Hypothèses du modèle** :
- Le sous-jacent suit un Mouvement Brownien Géométrique (GBM) : $dS = \mu S\, dt + \sigma S\, dW_t$
- Pas de dividendes, pas de coûts de transaction
- Taux sans risque $r$ constant, $\sigma$ constante
- Marché complet et continu (pas de sauts)

**EDP de Black-Scholes** (par argument d'absence d'arbitrage / delta-hedging) :

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

Le drift $\mu$ disparaît : c'est la neutralité au risque.

**Formule fermée — Call européen** :

$$C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

$\Phi$ : CDF de la loi normale standard. $\Phi(d_2)$ = probabilité risk-neutral d'exercice. $S_0\Phi(d_1)$ = espérance actualisée du sous-jacent conditionnée à l'exercice.

**Put-Call Parity** (modèle-indépendante) :
$$C - P = S_0 - Ke^{-rT}$$

---

## Insights Forums

- **"Dérivez l'EDP BS."** — Construire un portefeuille $\Pi = V - \Delta S$ delta-hedgé, appliquer Itô à $V(S,t)$, choisir $\Delta = \partial V/\partial S$ pour annuler le risque, puis appliquer l'absence d'arbitrage ($d\Pi = r\Pi\, dt$).
- **"Que représente $\Phi(d_2)$ ?"** — La probabilité risk-neutral $\mathbb{Q}(S_T > K)$. Et $\Phi(d_1)$ est la probabilité sous la mesure de l'action (mesure forward du sous-jacent).
- **"Pourquoi $\mu$ n'apparaît pas dans la formule BS ?"** — Le delta-hedging élimine le risque directionnel. La valorisation est obtenue par réplication, indépendamment des préférences des agents (théorème de Feynman-Kac).
- **"Quel est le prix d'un call quand $\sigma \to 0$ ?"** — $\max(S_0 e^{rT} - K, 0) \cdot e^{-rT} = \max(S_0 - Ke^{-rT}, 0)$ : la valeur actualisée du payoff déterministe.
- **"BS sur un actif qui verse des dividendes continus $q$ ?"** — Remplacer $S_0$ par $S_0 e^{-qT}$ et $r$ par $r - q$ dans $d_1$.

---

## Analyse des Limites

| Hypothèse violée | Impact réel |
|---|---|
| **$\sigma$ constante** | En pratique : smile/skew de volatilité implicite. Black-Scholes ne peut pas le reproduire sans extension (Heston, SABR). |
| **Marché continu** | Les sauts (crashes) créent des risques non-couverts. Un delta-hedge discret laisse un risque résiduel ($\Gamma$ risk). |
| **Liquidité infinie** | Le large delta-hedging d'un desk impacte les prix (market impact). L'hypothèse d'absence d'impact est irréaliste sur les illiquid assets. |
| **$r$ constant** | Pour les options à long terme (LEAPS, swaptions), les taux stochastiques (Vasicek, Hull-White) sont nécessaires. |
| **Pas de coûts de transaction** | La couverture continue est infiniment coûteuse en pratique (Leland 1985 : ajuster $\sigma$ pour intégrer les coûts). |
| **Distribution log-normale** | Les queues réelles sont épaisses (fat tails). BS sous-price les options très OTM et très ITM. |

---

## Contre-intuitions (Pièges recruteurs)

1. **"Plus la vol est élevée, moins vaut le call."** — Faux. La volatilité augmente toujours la valeur d'une option (le $\mathcal{V}ega > 0$). Une vol plus élevée donne plus de chances d'upside, sans risque supplémentaire pour un acheteur de call (asymétrie de payoff).

2. **"Un call deep-ITM vaut $S - Ke^{-rT}$ exactement."** — Presque vrai ($d_1, d_2 \to +\infty$ donc $\Phi \to 1$), mais on a encore une valeur temps résiduelle si $T > 0$.

3. **"BS donne le prix 'juste' d'une option."** — Non. BS donne un prix de non-arbitrage sous ses hypothèses. C'est un modèle de conversion vol implicite ↔ prix. Le marché cite les options en volatilité implicite.

4. **"Delta d'un call est $\Phi(d_1)$ donc il est toujours entre 0 et 1."** — Vrai pour un call vanille européen. Mais delta peut dépasser 1 pour des options barrière (e.g., call up-and-out près de la barrière).

5. **"La put-call parity est une conséquence de BS."** — Faux. La PCP est un résultat d'absence d'arbitrage pur, valable pour tout modèle (même avec stochastic vol ou jumps), du moment que les marchés sont sans friction.
