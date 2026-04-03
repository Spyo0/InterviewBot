# Fiche de Maîtrise : Lemme d'Itô

## Théorie

Le Lemme d'Itô est la règle de changement de variable pour les processus stochastiques. Il généralise la règle de la chaîne du calcul ordinaire au cadre du mouvement brownien.

**Mouvement brownien standard** $W_t$ :
- $dW_t \sim \mathcal{N}(0, dt)$
- Propriété clé : $(dW_t)^2 = dt$ (règle quadratique)

**Processus d'Itô général** :
$$dX_t = \mu(X_t, t)\, dt + \sigma(X_t, t)\, dW_t$$

**Énoncé du Lemme d'Itô** : Pour $f(X_t, t) \in C^{2,1}$,

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial X} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial X^2} (dX_t)^2$$

En substituant $(dX_t)^2 = \sigma^2 dt$ :

$$df = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial X} + \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial X^2}\right)dt + \sigma \frac{\partial f}{\partial X} dW_t$$

**Application au GBM** ($dS = \mu S\, dt + \sigma S\, dW_t$, $f = \ln S$) :

$$d(\ln S_t) = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma\, dW_t$$

Donc $\ln S_T \sim \mathcal{N}\!\left(\ln S_0 + \left(\mu - \frac{\sigma^2}{2}\right)T,\ \sigma^2 T\right)$.

---

## Insights Forums (Questions d'entretien fréquentes)

- **"Dérivez $d(\ln S)$ à partir du GBM."** — La question test fondamentale. Erreur classique : oublier le terme $-\frac{\sigma^2}{2}$.
- **"Pourquoi le terme d'Itô $\frac{1}{2}\sigma^2$ apparaît-il ?"** — À cause de $(dW)^2 = dt$ : le développement de Taylor stochastique doit aller à l'ordre 2.
- **"Calculez $\mathbb{E}[S_T]$ et expliquez pourquoi $\mathbb{E}[\ln S_T] \neq \ln \mathbb{E}[S_T]$."** — Jensen : $\ln$ est concave, donc $\mathbb{E}[\ln X] < \ln \mathbb{E}[X]$.
- **"Qu'est-ce qu'une intégrale d'Itô vs une intégrale de Stratonovich ?"** — L'intégrale d'Itô évalue l'intégrande au début de chaque intervalle (non-anticipante). Stratonovich au milieu (correction $+\frac{1}{2}\sigma\frac{\partial\sigma}{\partial X}$).

---

## Analyse des Limites

| Scénario | Problème |
|---|---|
| **Sauts de prix** | Le Lemme d'Itô s'applique aux processus continus. En présence de sauts (Poisson), il faut le Lemme d'Itô généralisé avec terme de saut $\Delta f$. |
| **Volatilité stochastique** | Si $\sigma = \sigma(t, W_t)$, le calcul tient, mais $\sigma$ est elle-même un processus. Heston requiert un système d'EDPs couplées. |
| **Temps discret** | En pratique les données sont discrètes : l'approximation $\ln(S_{t+\Delta t}/S_t) \approx (\mu - \sigma^2/2)\Delta t + \sigma\epsilon\sqrt{\Delta t}$ introduit un biais de discrétisation. |
| **$f$ non $C^2$** | Si la payoff est discontinue (e.g., option digitale), $\frac{\partial^2 f}{\partial S^2}$ n'existe pas au sens classique : il faut la théorie des distributions. |

---

## Contre-intuitions (Pièges recruteurs)

1. **"Le drift de $\ln S$ est $\mu$, non ?"** — Non. C'est $\mu - \frac{\sigma^2}{2}$. La correction est due à la convexité de $\ln$. Le drift arithmétique et le drift géométrique diffèrent.

2. **"Si $dX = \sigma\, dW$, alors $\mathbb{E}[X_T] = X_0$, donc l'espérance de $e^{X_T}$ est $e^{X_0}$."** — Faux. $\mathbb{E}[e^{X_T}] = e^{X_0 + \frac{1}{2}\sigma^2 T}$ (par la formule moment-génératrice de la loi normale). La convexité de l'exponentielle crée un gain systématique.

3. **"Le Lemme d'Itô s'applique à tout processus stochastique."** — Non. Il requiert que le processus soit une semi-martingale (en particulier, des trajectoires à variation quadratique finie). Un processus fractionnaire avec $H \neq 1/2$ nécessite un calcul fractionnaire différent.

4. **"La correction $-\sigma^2/2$ disparaît en monde risk-neutral car on remplace $\mu$ par $r$."** — La correction reste : $d(\ln S) = (r - \frac{\sigma^2}{2})dt + \sigma dW^Q$. Elle ne dépend pas du drift.
