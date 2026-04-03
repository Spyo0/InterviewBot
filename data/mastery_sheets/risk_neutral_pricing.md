# Fiche de Maîtrise : Pricing Risk-Neutral & Changement de Mesure

## Théorie

**Principe fondamental** : Dans un marché sans arbitrage et complet, il existe une unique mesure de probabilité $\mathbb{Q}$ (mesure risk-neutral / martingale équivalente) sous laquelle les prix actualisés des actifs sont des martingales.

**Mesure physique $\mathbb{P}$ vs mesure risk-neutral $\mathbb{Q}$** :

Sous $\mathbb{P}$ : $dS = \mu S\, dt + \sigma S\, dW_t^{\mathbb{P}}$

Sous $\mathbb{Q}$ : $dS = r S\, dt + \sigma S\, dW_t^{\mathbb{Q}}$

**Processus de Girsanov** — Changement de mesure :
$$dW_t^{\mathbb{Q}} = dW_t^{\mathbb{P}} + \theta\, dt, \quad \theta = \frac{\mu - r}{\sigma}$$

$\theta$ est le **prix du risque de marché** (Sharpe ratio instantané).

**Densité de Radon-Nikodym** :
$$\frac{d\mathbb{Q}}{d\mathbb{P}}\bigg|_{\mathcal{F}_T} = \exp\!\left(-\theta W_T^{\mathbb{P}} - \frac{1}{2}\theta^2 T\right)$$

**Théorème de Feynman-Kac** : Le prix d'un dérivé est l'espérance actualisée sous $\mathbb{Q}$ :
$$V(S, t) = e^{-r(T-t)}\,\mathbb{E}^{\mathbb{Q}}\!\left[f(S_T) \mid \mathcal{F}_t\right]$$

Ce qui est équivalent à résoudre l'EDP de Black-Scholes.

**Mesures alternatives** :
- **Mesure forward $\mathbb{Q}^T$** (numéraire = ZCB $P(t,T)$) : simplifie le pricing des taux d'intérêt.
- **Mesure de l'action $\mathbb{Q}^S$** (numéraire = $S_t$) : $d_1$ dans BS est $\mathbb{Q}^S(S_T > K)$.

---

## Insights Forums

- **"Pourquoi le drift $\mu$ disparaît dans le pricing ?"** — Car le portefeuille de réplication annule le risque directionnel : les préférences des agents n'entrent pas en jeu. C'est le "magic" du delta-hedging complet.
- **"Expliquez le théorème de Girsanov en une phrase."** — Girsanov dit qu'on peut transformer un drift brownien quelconque en un brownien standard en changeant de mesure de probabilité, à condition que la densité de Radon-Nikodym soit une martingale (condition de Novikov).
- **"Qu'est-ce qu'une martingale ?"** — Un processus $M_t$ tel que $\mathbb{E}[M_T | \mathcal{F}_t] = M_t$ : la meilleure prédiction de la valeur future est la valeur présente. Les prix actualisés sont des martingales sous $\mathbb{Q}$.
- **"Peut-on pricer n'importe quel dérivé avec une mesure risk-neutral ?"** — Uniquement dans un marché complet (chaque risque est couvrable). Pour les marchés incomplets (jumps, vol stochastique, risques non-tradables), il y a une infinité de mesures $\mathbb{Q}$ : le prix est une fourchette, pas un point.

---

## Analyse des Limites

| Hypothèse | Limite |
|---|---|
| **Marché complet** | En présence de vol stochastique (Heston), le risque de vol n'est pas couvrable avec seulement le sous-jacent. Le prix du risque de vol $\lambda$ doit être spécifié : prix non-unique. |
| **Pas de sauts** | Les sauts (processus de Lévy) créent des risques non-couverts. La mesure $\mathbb{Q}$ reste libre sur l'intensité et la distribution des sauts. |
| **Liquidité** | La réplication continue suppose une liquidité infinie. En pratique, l'impact de marché et les coûts de transaction rendent la réplication imparfaite. |
| **Taux déterministe** | Pour les dérivés de taux, $r$ est lui-même stochastique. Le numéraire doit être choisi avec soin (compte bancaire vs ZCB). |
| **Condition de Novikov** | La densité de Girsanov doit être une martingale ($\mathbb{E}[e^{\frac{1}{2}\int_0^T \theta_s^2 ds}] < \infty$). Violée pour certains modèles exotiques. |

---

## Contre-intuitions (Pièges recruteurs)

1. **"Sous $\mathbb{Q}$, tous les actifs ont le même rendement $r$."** — Vrai pour les actifs tradables. Les actifs non-tradables (vol, corrélation) n'ont pas de rendement fixé sous $\mathbb{Q}$ : leur drift dépend du prix du risque spécifié.

2. **"La mesure risk-neutral $\mathbb{Q}$ est la vraie probabilité du marché."** — Faux. C'est une construction mathématique pour le pricing. Sous $\mathbb{Q}$, les actifs performent en moyenne comme le taux sans risque, ce qui est irréaliste ($\mu \neq r$ en général).

3. **"Feynman-Kac et Monte Carlo sont deux méthodes différentes."** — Ils sont la même chose sous deux angles : Feynman-Kac dit que la solution de l'EDP est une espérance, et MC calcule cette espérance numériquement.

4. **"Le prix du risque $\theta$ est connu."** — Il est calibré aux prix de marché. En modèle BS, $\theta = (\mu - r)/\sigma$, mais $\mu$ (drift réel) n'est pas observable de façon fiable. C'est pourquoi on calibre sur la vol implicite plutôt que la vol historique.

5. **"En marché incomplet, le modèle ne donne pas de prix."** — Il donne une fourchette de prix sans-arbitrage (super/sous-réplication). En pratique, on choisit une mesure $\mathbb{Q}$ via calibration ou critère d'entropie minimale, et on obtient un prix "préféré".
