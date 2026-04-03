"""Catalogue de fiches de révision pour l'onglet Cours."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CourseSheet:
    slug: str
    title: str
    icon: str
    category: str
    level: str
    duration: str
    tags: tuple[str, ...]
    summary: str
    content: str


@dataclass(frozen=True)
class CourseTrack:
    slug: str
    title: str
    description: str
    topics: tuple[str, ...]


def _bullet_section(title: str, items: list[str]) -> str:
    return f"### {title}\n" + "\n".join(f"- {item}" for item in items)


def _formula_section(formulas: list[str]) -> str:
    return "### Formules et relations\n" + "\n".join(f"- {formula}" for formula in formulas)


def _build_sheet(
    *,
    slug: str,
    title: str,
    icon: str,
    category: str,
    level: str,
    duration: str,
    tags: tuple[str, ...],
    summary: str,
    intuition: list[str],
    formulas: list[str],
    pitfalls: list[str],
    interview_drills: list[str],
    signals: list[str],
) -> CourseSheet:
    content = "\n\n".join([
        _bullet_section("Intuition de travail", intuition),
        _formula_section(formulas),
        _bullet_section("Pièges classiques", pitfalls),
        _bullet_section("Angles d'entretien", interview_drills),
        _bullet_section("Ce qu'un bon candidat doit verbaliser", signals),
    ])
    return CourseSheet(
        slug=slug,
        title=title,
        icon=icon,
        category=category,
        level=level,
        duration=duration,
        tags=tags,
        summary=summary,
        content=content,
    )


COURSE_CATALOG: tuple[CourseSheet, ...] = (
    _build_sheet(
        slug="brownian_motion",
        title="Mouvement brownien",
        icon="W",
        category="Calcul stochastique",
        level="Fondamental",
        duration="8 min",
        tags=("processus", "martingales", "stochastique"),
        summary="Base de presque tout le pricing diffusion : accroissements indépendants, variance linéaire en temps, trajectoires continues et partout non dérivables.",
        intuition=[
            "Le brownien est le modèle canonique de bruit continu en temps.",
            "Sa moyenne reste nulle, mais sa variance croît comme le temps écoulé.",
            "On l'utilise surtout via ses incréments : $W_{t+h} - W_t \\sim \\mathcal{N}(0, h)$.",
        ],
        formulas=[
            "$W_0 = 0$.",
            "$W_t - W_s \\sim \\mathcal{N}(0, t-s)$ pour $t > s$.",
            "$\\mathbb{E}[W_t] = 0$ et $\\mathrm{Var}(W_t) = t$.",
            "$\\langle W \\rangle_t = t$.",
        ],
        pitfalls=[
            "Confondre trajectoire continue et trajectoire dérivable.",
            "Dire que les incréments sont indépendants sans préciser les intervalles disjoints.",
            "Oublier que la variance dépend du temps et non d'une constante.",
        ],
        interview_drills=[
            "Pourquoi le brownien n'est-il pas différentiable presque sûrement ?",
            "Que devient $W_t / t$ quand $t \\to \\infty$ ?",
            "Quelle différence conceptuelle entre brownien et marche aléatoire ?",
        ],
        signals=[
            "La volatilité d'un GBM vient du terme $\\sigma dW_t$.",
            "Le brownien est une martingale continue à variation quadratique finie.",
            "Le bon objet pour les calculs est l'incrément, pas la 'pente' de la trajectoire.",
        ],
    ),
    _build_sheet(
        slug="ito_lemma",
        title="Lemme d'Itô",
        icon="d",
        category="Calcul stochastique",
        level="Intermédiaire",
        duration="10 min",
        tags=("ito", "diffusions", "chain-rule"),
        summary="La règle de chaîne du monde stochastique : on ajoute le terme en dérivée seconde à cause de la variation quadratique.",
        intuition=[
            "Le lemme d'Itô remplace la règle de chaîne classique pour une fonction d'un processus diffusif.",
            "Le terme supplémentaire vient du fait que $(dW_t)^2 = dt$ au premier ordre utile.",
            "Il sert partout : Black-Scholes, changement de variables, calcul de dynamiques transformées.",
        ],
        formulas=[
            "Si $dX_t = a_t dt + b_t dW_t$, alors $df(t, X_t) = \\left(\\partial_t f + a_t \\partial_x f + \\tfrac{1}{2} b_t^2 \\partial_{xx} f\\right)dt + b_t \\partial_x f \\, dW_t$.",
            "Règles mnémotechniques : $(dt)^2 = 0$, $dt \\, dW_t = 0$, $(dW_t)^2 = dt$.",
        ],
        pitfalls=[
            "Oublier le terme en dérivée seconde.",
            "Mettre un mauvais coefficient devant $\\partial_{xx} f$.",
            "Appliquer Itô à une fonction non suffisamment régulière sans le signaler.",
        ],
        interview_drills=[
            "Appliquer Itô à $\\log S_t$ si $dS_t = \\mu S_t dt + \\sigma S_t dW_t$.",
            "Pourquoi $\\log S_t$ fait apparaître $-\\tfrac{1}{2}\\sigma^2$ ?",
            "Comment Itô permet-il d'identifier une martingale ?",
        ],
        signals=[
            "Le candidat doit verbaliser la variation quadratique et pas juste 'il y a un terme en plus'.",
            "Un bon réflexe est de factoriser les dérivées partielles proprement avant substitution.",
            "Il faut savoir passer de la dynamique de $S_t$ à celle de $f(S_t)$ sans improviser.",
        ],
    ),
    _build_sheet(
        slug="poisson_processes",
        title="Processus de Poisson",
        icon="N",
        category="Processus",
        level="Fondamental",
        duration="7 min",
        tags=("sauts", "intensité", "comptage"),
        summary="Le modèle standard de comptage aléatoire : indépendance des incréments, intensité constante, temps d'attente exponentiels.",
        intuition=[
            "Un processus de Poisson compte des événements rares qui arrivent à taux moyen constant.",
            "Sur un petit intervalle $dt$, on observe 0 ou 1 saut avec probabilité dominante.",
            "Le temps entre deux sauts suit une loi exponentielle.",
        ],
        formulas=[
            "$N_t \\sim \\mathrm{Poisson}(\\lambda t)$.",
            "$\\mathbb{E}[N_t] = \\lambda t$ et $\\mathrm{Var}(N_t) = \\lambda t$.",
            "$\\mathbb{P}(N_{t+dt} - N_t = 1) = \\lambda dt + o(dt)$.",
            "$\\mathbb{P}(\\tau_1 > t) = e^{-\\lambda t}$ pour le premier temps de saut.",
        ],
        pitfalls=[
            "Confondre intensité et probabilité de saut exacte.",
            "Oublier que plusieurs sauts sur $dt$ sont d'ordre $o(dt)$.",
            "Dire que les temps d'attente sont gaussiens ou uniformes.",
        ],
        interview_drills=[
            "Comment simuler un processus de Poisson ?",
            "Quelle est la différence entre un brownien et un processus de Poisson ?",
            "Comment passer à une intensité dépendante du temps ?",
        ],
        signals=[
            "Le candidat doit savoir relier la loi de comptage et les temps inter-arrivées.",
            "Il faut distinguer processus de comptage et taille des sauts.",
            "Un bon niveau mentionne la compensation $N_t - \\lambda t$.",
        ],
    ),
    _build_sheet(
        slug="futures_forwards",
        title="Forwards et Futures",
        icon="F",
        category="Produits",
        level="Fondamental",
        duration="8 min",
        tags=("forwards", "futures", "carry"),
        summary="Comprendre la logique de carry, la différence OTC / exchange-traded et l'effet du marking-to-market.",
        intuition=[
            "Un forward est un contrat OTC réglé à maturité ; un future est standardisé et margé quotidiennement.",
            "Le prix forward vient d'un argument de non-arbitrage : spot, financement, revenus et coûts de portage.",
            "Le future diffère du forward quand les corrélations taux / sous-jacent comptent réellement.",
        ],
        formulas=[
            "Sans revenus : $F_0 = S_0 e^{rT}$.",
            "Avec rendement continu $q$ : $F_0 = S_0 e^{(r-q)T}$.",
            "Avec coût de stockage $u$ : $F_0 = S_0 e^{(r+u)T}$.",
        ],
        pitfalls=[
            "Dire que forward et future sont toujours identiques.",
            "Oublier les dividendes, coupons ou convenience yield.",
            "Parler de 'prix futur attendu' au lieu de prix d'arbitrage.",
        ],
        interview_drills=[
            "Quand un future sur taux peut-il s'écarter d'un forward ?",
            "Quel arbitrage met-on en place si le future est trop cher ?",
            "Comment intégrer un convenience yield sur une commodité ?",
        ],
        signals=[
            "Le candidat doit relier le prix à un portefeuille cash-and-carry.",
            "Il faut parler explicitement du marking-to-market pour les futures.",
            "Le bon raisonnement reste financier avant d'être purement algébrique.",
        ],
    ),
    _build_sheet(
        slug="risk_neutral_pricing",
        title="Pricing risk-neutral",
        icon="Q",
        category="Pricing",
        level="Intermédiaire",
        duration="10 min",
        tags=("measure", "martingale", "pricing"),
        summary="Sous la mesure risque-neutre, les actifs actualisés deviennent des martingales et les prix se lisent comme des espérances actualisées.",
        intuition=[
            "La mesure risque-neutre n'affirme pas que le monde réel est neutre au risque.",
            "C'est un outil de pricing construit par absence d'arbitrage.",
            "Le drift pertinent du sous-jacent devient le taux sans risque ajusté des revenus de portage.",
        ],
        formulas=[
            "$V_0 = e^{-rT} \\mathbb{E}^{\\mathbb{Q}}[\\Phi(S_T)]$.",
            "Sous $\\mathbb{Q}$, pour une action à dividende continu $q$ : $dS_t = (r-q)S_t dt + \\sigma S_t dW_t^{\\mathbb{Q}}$.",
            "Un actif actualisé sans arbitrage est une martingale sous une mesure équivalente appropriée.",
        ],
        pitfalls=[
            "Confondre probabilité historique et probabilité de pricing.",
            "Dire qu'on 'supprime le risque' en changeant de mesure.",
            "Oublier le rôle du numéraire.",
        ],
        interview_drills=[
            "Pourquoi a-t-on le droit de remplacer le drift historique par $r$ ?",
            "Que devient le pricing si le taux est stochastique ?",
            "Quel est le lien entre measure change et Girsanov ?",
        ],
        signals=[
            "Le candidat doit articuler absence d'arbitrage, numéraire et martingale.",
            "Il doit éviter une réponse purement mnémotechnique du type 'on remplace $\\mu$ par $r$'.",
            "Un bon niveau mentionne que la mesure dépend du numéraire choisi.",
        ],
    ),
    _build_sheet(
        slug="binomial_trees",
        title="Arbres binomiaux",
        icon="T",
        category="Pricing",
        level="Fondamental",
        duration="8 min",
        tags=("discret", "replication", "american"),
        summary="Le laboratoire discret de l'absence d'arbitrage : réplication locale, probabilité risque-neutre et gestion naturelle de l'exercice américain.",
        intuition=[
            "L'arbre binomial approxime une diffusion par des mouvements up/down sur des pas discrets.",
            "Le prix vient d'une réplication locale ou d'une espérance sous la probabilité risque-neutre.",
            "L'exercice américain se gère facilement par backward induction.",
        ],
        formulas=[
            "$S_{t+\\Delta t} \\in \\{uS_t, dS_t\\}$.",
            "$p^* = \\frac{e^{r\\Delta t} - d}{u-d}$.",
            "$V_t = e^{-r\\Delta t}(p^* V_u + (1-p^*)V_d)$.",
        ],
        pitfalls=[
            "Choisir des paramètres $u, d$ incohérents avec l'absence d'arbitrage.",
            "Oublier la comparaison exercice immédiat / continuation pour une américaine.",
            "Dire que la probabilité risque-neutre est une vraie fréquence empirique.",
        ],
        interview_drills=[
            "Pourquoi une call américaine sans dividende ne s'exerce-t-elle pas tôt ?",
            "Comment choisir $u$ et $d$ pour converger vers Black-Scholes ?",
            "Quel est l'intérêt d'un arbre trinomial ?",
        ],
        signals=[
            "Le bon candidat explique la backward induction sans se perdre dans les notations.",
            "Il sait relier arbre binomial et réplication locale.",
            "Il comprend quand le discret est plus pratique que la formule fermée.",
        ],
    ),
    _build_sheet(
        slug="black_scholes",
        title="Black-Scholes",
        icon="B",
        category="Pricing",
        level="Intermédiaire",
        duration="12 min",
        tags=("bsm", "closed form", "hedging"),
        summary="Le modèle central du pricing optionnel en diffusion lognormale, à comprendre comme un résultat de réplication autant qu'une formule.",
        intuition=[
            "Black-Scholes n'est pas juste une formule : c'est une conséquence d'un portefeuille auto-finançant et sans arbitrage.",
            "La call européenne s'obtient en neutralisant le risque diffusif par delta-hedging.",
            "La fermeture analytique vient de la lognormalité de $S_T$ sous $\\mathbb{Q}$.",
        ],
        formulas=[
            "$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$.",
            "$d_1 = \\frac{\\ln(S_0/K) + (r-q+\\tfrac{1}{2}\\sigma^2)T}{\\sigma\\sqrt{T}}$.",
            "$d_2 = d_1 - \\sigma\\sqrt{T}$.",
            "$\\Delta_{call} = e^{-qT}N(d_1)$.",
        ],
        pitfalls=[
            "Réciter la formule sans expliquer la réplication derrière.",
            "Confondre $d_1$ et $d_2$ dans l'interprétation.",
            "Oublier l'impact des dividendes continus.",
        ],
        interview_drills=[
            "Pourquoi $d_1$ intervient dans le delta alors que $d_2$ intervient dans l'exercice ?",
            "Quelles hypothèses du modèle cassent d'abord en pratique ?",
            "Comment retrouver Black-Scholes à partir de la PDE ?",
        ],
        signals=[
            "Le candidat doit passer sans douleur de la PDE à la formule ou inversement.",
            "Il doit parler de hedge continu et de limites pratiques de cette hypothèse.",
            "Une bonne réponse relie prix, couverture et mesure risque-neutre.",
        ],
    ),
    _build_sheet(
        slug="greeks",
        title="Grecques",
        icon="Δ",
        category="Gestion du risque",
        level="Intermédiaire",
        duration="10 min",
        tags=("delta", "gamma", "vega", "theta"),
        summary="Les sensibilités sont le langage du desk : elles condensent le comportement local du portefeuille face aux facteurs de risque.",
        intuition=[
            "Le delta mesure la sensibilité première au sous-jacent.",
            "Le gamma mesure la convexité : il dit comment le delta change.",
            "Le vega et le theta lient le prix à la volatilité implicite et au passage du temps.",
        ],
        formulas=[
            "$\\Delta = \\partial_S V$.",
            "$\\Gamma = \\partial_{SS} V$.",
            "$\\mathcal{V} = \\partial_\\sigma V$.",
            "$\\Theta = \\partial_t V$.",
        ],
        pitfalls=[
            "Parler d'un seul greek isolé sans discuter les interactions.",
            "Confondre sensibilité locale et P&L réellement réalisé sur grand mouvement.",
            "Oublier le rôle du gamma dans le coût de couverture.",
        ],
        interview_drills=[
            "Pourquoi un portefeuille long gamma est souvent short theta ?",
            "Quel risque reste-t-il après delta-hedging ?",
            "Comment le smile déforme l'usage du vega ?",
        ],
        signals=[
            "Il faut relier les greeks à la gestion quotidienne du book.",
            "Le bon candidat parle en P&L : delta move, gamma scalp, theta bleed.",
            "Il distingue la sensibilité modèle et le risque de recalibration.",
        ],
    ),
    _build_sheet(
        slug="implied_volatility",
        title="Volatilité implicite et smile",
        icon="σ",
        category="Volatilité",
        level="Intermédiaire",
        duration="10 min",
        tags=("smile", "surface", "vol"),
        summary="La vol implicite est le prix de marché reformulé en paramètre de modèle ; le smile encode ce que Black-Scholes ne sait pas capturer seul.",
        intuition=[
            "On inverse Black-Scholes pour traduire un prix observé en volatilité implicite.",
            "La surface dépend du strike et de la maturité, pas d'un seul scalaire.",
            "Le smile reflète asymétries, queues épaisses, sauts et contraintes d'offre/demande.",
        ],
        formulas=[
            "La vol implicite est la solution de $BS(\\sigma_{impl}) = P_{marché}$.",
            "Le skew equity est typiquement décroissant en strike.",
            "La surface doit respecter des contraintes d'absence d'arbitrage en strike et maturité.",
        ],
        pitfalls=[
            "Dire que la vol implicite est une vraie volatilité future attendue au sens strict.",
            "Oublier les arbitrages calendrier ou butterfly.",
            "Parler du smile sans distinguer smile et skew selon les classes d'actifs.",
        ],
        interview_drills=[
            "Pourquoi les equities ont-elles souvent un skew négatif ?",
            "Comment lire une surface en pratique ?",
            "Quelle différence entre vol historique et vol implicite ?",
        ],
        signals=[
            "Le candidat doit comprendre qu'une surface est un objet de marché, pas seulement un input de modèle.",
            "Il doit savoir relier smile et couverture de risques extrêmes.",
            "Un bon niveau mentionne l'absence d'arbitrage statique.",
        ],
    ),
    _build_sheet(
        slug="local_vs_stochastic_vol",
        title="Vol locale vs vol stochastique",
        icon="V",
        category="Volatilité",
        level="Élevé",
        duration="12 min",
        tags=("dupire", "heston", "surface"),
        summary="Deux façons très différentes de réconcilier modèle et surface : calibration parfaite instantanée pour la vol locale, dynamique plus réaliste pour la vol stochastique.",
        intuition=[
            "La vol locale colle exactement à une surface du jour, mais sa dynamique future peut être peu réaliste.",
            "La vol stochastique introduit un facteur latent de variance et améliore la dynamique du smile.",
            "Le bon modèle dépend de l'usage : pricing vanilla, exotiques, hedging de surface.",
        ],
        formulas=[
            "Vol locale : $dS_t = (r-q)S_t dt + \\sigma_{loc}(t, S_t) S_t dW_t$.",
            "Exemple Heston : $dv_t = \\kappa(\\theta-v_t)dt + \\xi\\sqrt{v_t}dZ_t$ avec corrélation $\\rho$.",
            "Dupire relie la surface call à $\\sigma_{loc}(t, K)$.",
        ],
        pitfalls=[
            "Dire qu'un modèle qui calibre mieux est automatiquement meilleur en hedging.",
            "Oublier la dynamique de smile forward.",
            "Comparer les modèles uniquement sur la qualité du fit vanilla.",
        ],
        interview_drills=[
            "Pourquoi la vol locale peut-elle mal hedger certains exotiques ?",
            "Que change la corrélation spot-vol dans Heston ?",
            "Comment comparer local vol et stochastic vol sur un desk ?",
        ],
        signals=[
            "Le candidat doit comparer fit instantané et réalisme dynamique.",
            "Il doit savoir parler des smiles forward et de la stabilité de calibration.",
            "Une réponse senior évoque usage desk et gouvernance modèle.",
        ],
    ),
    _build_sheet(
        slug="monte_carlo",
        title="Monte Carlo",
        icon="MC",
        category="Simulation",
        level="Intermédiaire",
        duration="10 min",
        tags=("simulation", "variance reduction", "path-dependent"),
        summary="Le marteau de tous les payoffs path-dependent, à condition de savoir contrôler le bruit statistique et le coût de calcul.",
        intuition=[
            "Monte Carlo estime une espérance par moyenne empirique de scénarios simulés.",
            "La précision décroît lentement : l'erreur est en ordre $1/\\sqrt{N}$.",
            "L'essentiel en entretien est souvent la réduction de variance, pas la boucle for.",
        ],
        formulas=[
            "$\\hat{V}_0 = e^{-rT}\\frac{1}{N}\\sum_{i=1}^N \\Phi(S_T^{(i)})$.",
            "Erreur standard en ordre $\\sigma_\\Phi / \\sqrt{N}$.",
            "Exemples de variance reduction : antithétiques, control variates, stratification.",
        ],
        pitfalls=[
            "Penser que doubler la précision coûte deux fois plus cher au lieu de quatre.",
            "Oublier la discrétisation temporelle pour les payoffs path-dependent.",
            "Négliger le biais de schéma numérique en se focalisant seulement sur la variance.",
        ],
        interview_drills=[
            "Comment pricer une asian option par Monte Carlo ?",
            "Pourquoi une control variate bien choisie est-elle si puissante ?",
            "Quand préférer PDE ou arbre à Monte Carlo ?",
        ],
        signals=[
            "Le candidat doit séparer erreur statistique et erreur de discrétisation.",
            "Il doit savoir citer des techniques de variance reduction concrètes.",
            "Une bonne réponse relie méthode numérique et nature du payoff.",
        ],
    ),
    _build_sheet(
        slug="interest_rate_curves",
        title="Courbes de taux et forwards",
        icon="R",
        category="Taux",
        level="Intermédiaire",
        duration="9 min",
        tags=("discounting", "curve", "forward rates"),
        summary="La courbe donne les prix du temps : discount factors, zero rates, forwards implicites et cohérence d'ensemble.",
        intuition=[
            "On construit une courbe pour actualiser et projeter des flux futurs de manière cohérente.",
            "Un taux forward n'est pas une prévision naïve, c'est un taux implicite sans arbitrage.",
            "En pratique, plusieurs courbes coexistent : discounting OIS, projection IBOR ou indices dérivés.",
        ],
        formulas=[
            "$P(0,T)$ : discount factor à maturité $T$.",
            "$R(0,T) = -\\frac{1}{T}\\ln P(0,T)$ en zéro-coupon continu.",
            "$f(0; T_1, T_2) = \\frac{1}{T_2-T_1}\\ln\\left(\\frac{P(0,T_1)}{P(0,T_2)}\\right)$ en continu.",
        ],
        pitfalls=[
            "Confondre courbe de discount et courbe de projection.",
            "Parler des forwards comme de simples anticipations macro.",
            "Oublier que le bootstrapping dépend des instruments observés.",
        ],
        interview_drills=[
            "Comment extraire un forward 6m6m d'une courbe ?",
            "Pourquoi le multi-curve est-il devenu la norme ?",
            "Quel impact d'une courbe mal construite sur un pricing swaption ?",
        ],
        signals=[
            "Le candidat doit penser en discount factors avant de penser en taux.",
            "Il doit relier la courbe aux conventions de marché réelles.",
            "Une réponse solide mentionne le bootstrapping et les instruments d'ancrage.",
        ],
    ),
    _build_sheet(
        slug="pde_pricing",
        title="PDE de pricing",
        icon="∂",
        category="Pricing",
        level="Élevé",
        duration="12 min",
        tags=("pde", "replication", "boundary conditions"),
        summary="La PDE est la version continue de la réplication : on élimine le risque diffusif et on impose l'absence d'arbitrage local.",
        intuition=[
            "La PDE naît du delta-hedging d'un portefeuille auto-finançant.",
            "Elle donne un cadre général bien au-delà des seules solutions fermées.",
            "Les conditions terminales et aux bords sont aussi importantes que l'équation elle-même.",
        ],
        formulas=[
            "Pour Black-Scholes : $\\partial_t V + \\tfrac{1}{2}\\sigma^2 S^2 \\partial_{SS}V + (r-q)S\\partial_S V - rV = 0$.",
            "Condition terminale pour une call : $V(T,S) = (S-K)^+$.",
            "Conditions aux bords typiques : comportement quand $S \\to 0$ ou $S \\to \\infty$.",
        ],
        pitfalls=[
            "Écrire la PDE sans expliquer le portefeuille de couverture.",
            "Oublier les conditions terminales ou de bord.",
            "Confondre probabilité neutre au risque et simple substitution algébrique.",
        ],
        interview_drills=[
            "Comment justifier la PDE sans appeler directement Feynman-Kac ?",
            "Pourquoi les conditions aux bords comptent-elles autant ?",
            "Quand une PDE est-elle préférable à Monte Carlo ?",
        ],
        signals=[
            "Le candidat doit savoir raconter la logique hedge -> portefeuille sans risque -> rendement au taux sans risque.",
            "Une bonne réponse relie PDE, espérance sous $\\mathbb{Q}$ et schémas numériques.",
            "Le niveau élevé se voit dans la maîtrise des conditions aux limites.",
        ],
    ),
    _build_sheet(
        slug="martingales_stopping",
        title="Martingales et optional stopping",
        icon="M",
        category="Calcul stochastique",
        level="Intermédiaire",
        duration="9 min",
        tags=("martingale", "stopping time", "filtration"),
        summary="Un sujet central en entretien : savoir reconnaître une martingale, comprendre ce qu'autorise un stopping time, et éviter les faux raisonnements de casino.",
        intuition=[
            "Une martingale modélise un processus dont la meilleure prévision future, conditionnellement à l'information courante, est sa valeur actuelle.",
            "Le théorème d'optional stopping ne dit pas qu'on peut gagner à tous les coups en arrêtant intelligemment.",
            "Les hypothèses de bornitude ou d'intégrabilité sont aussi importantes que l'identité finale.",
        ],
        formulas=[
            "$\\mathbb{E}[M_t \\mid \\mathcal{F}_s] = M_s$ pour $s \\le t$.",
            "Si $\\tau$ est un stopping time admissible et si les hypothèses tiennent, alors $\\mathbb{E}[M_\\tau] = \\mathbb{E}[M_0]$.",
            "Exemple classique : $W_t$ est une martingale, mais $W_t^2 - t$ aussi.",
        ],
        pitfalls=[
            "Dire qu'un processus de moyenne constante est automatiquement une martingale.",
            "Oublier la filtration et la notion d'adaptation.",
            "Citer optional stopping sans jamais rappeler les hypothèses qui rendent le résultat valide.",
        ],
        interview_drills=[
            "Pourquoi $W_t^2$ n'est-il pas une martingale ?",
            "Dans quel cas l'arrêt au premier passage casse-t-il le raisonnement naïf ?",
            "Comment distinguer submartingale, martingale et supermartingale ?",
        ],
        signals=[
            "Le candidat doit parler de conditionnement et d'information disponible.",
            "Une bonne réponse donne au moins un contre-exemple aux intuitions erronées.",
            "Le niveau supérieur se voit dans la maîtrise des conditions d'intégrabilité uniforme.",
        ],
    ),
    _build_sheet(
        slug="girsanov_numeraire",
        title="Girsanov et changement de numéraire",
        icon="G",
        category="Pricing",
        level="Élevé",
        duration="12 min",
        tags=("girsanov", "numeraire", "measure change"),
        summary="Le vrai niveau quant apparaît ici : comprendre pourquoi le drift change, ce que transporte Radon-Nikodym, et comment le numéraire simplifie certains pricings.",
        intuition=[
            "Girsanov permet de modifier la dérive d'un processus brownien en changeant de mesure de probabilité.",
            "Le changement de numéraire est souvent la manière la plus propre de rendre un prix actualisé martingale.",
            "Le résultat n'est pas 'magique' : on paie ce changement via une densité de probabilité.",
        ],
        formulas=[
            "Sous une nouvelle mesure $\\mathbb{Q}$, un brownien peut s'écrire $W_t^{\\mathbb{Q}} = W_t + \\int_0^t \\theta_s ds$.",
            "La densité de changement de mesure s'écrit typiquement sous forme d'exponentielle stochastique.",
            "Un actif divisé par son numéraire est martingale sous la mesure associée à ce numéraire.",
        ],
        pitfalls=[
            "Dire que Girsanov change la volatilité alors qu'il change la dérive.",
            "Oublier que toutes les mesures ne sont pas équivalentes sans conditions suffisantes.",
            "Parler du numéraire comme d'un simple facteur d'actualisation générique.",
        ],
        interview_drills=[
            "Pourquoi choisir le bond zéro-coupon comme numéraire sur un pricing de taux ?",
            "Quel lien entre forward measure et pricing caplet ?",
            "Que se passe-t-il si les conditions de Novikov ne sont pas satisfaites ?",
        ],
        signals=[
            "Le candidat doit relier changement de mesure, densité et drift ajusté.",
            "Une réponse solide distingue clairement mesure risque-neutre et forward measure.",
            "Le bon niveau sait expliquer l'intérêt pratique du choix de numéraire dans un pricing ciblé.",
        ],
    ),
    _build_sheet(
        slug="swaps_and_swap_rate",
        title="Swaps de taux et swap rate",
        icon="S",
        category="Taux",
        level="Fondamental",
        duration="9 min",
        tags=("swap", "annuity", "par rate"),
        summary="Indispensable en fixed income : jambes fixe / float, par swap rate, annuity et lecture économique d'un swap vanille.",
        intuition=[
            "Un swap échange des flux fixes contre des flux flottants sur un nominal donné.",
            "Le par swap rate est le taux fixe qui rend la valeur initiale du swap nulle.",
            "L'annuity mesure le poids actualisé de la jambe fixe.",
        ],
        formulas=[
            "$\\text{PV fixe} = K \\sum_i \\alpha_i P(0,T_i)$.",
            "$\\text{PV float}$ s'exprime à partir de la courbe de projection et de discount.",
            "$S_{swap}(0) = \\frac{\\text{PV float}}{\\sum_i \\alpha_i P(0,T_i)}$.",
        ],
        pitfalls=[
            "Confondre taux swap et moyenne simple des forwards.",
            "Oublier la convention de day count dans les accrual factors $\\alpha_i$.",
            "Parler d'une seule courbe en oubliant l'approche multi-curve.",
        ],
        interview_drills=[
            "Comment dériver le par swap rate à partir de discount factors ?",
            "Pourquoi la jambe flottante a-t-elle une formule plus compacte à la date de reset ?",
            "Quel impact d'une hausse parallèle de courbe sur un payeur fixe ?",
        ],
        signals=[
            "Le candidat doit savoir écrire l'annuity sans hésitation.",
            "Une bonne réponse relie le taux swap à une moyenne pondérée de forwards, pas à une moyenne naïve.",
            "Le niveau pro se voit dans la maîtrise des conventions et du multi-curve.",
        ],
    ),
    _build_sheet(
        slug="duration_convexity",
        title="Duration et convexité",
        icon="D",
        category="Taux",
        level="Fondamental",
        duration="8 min",
        tags=("bond", "duration", "convexity"),
        summary="Les sensibilités de base en fixed income : première approximation par la duration, correction non linéaire par la convexité.",
        intuition=[
            "La duration mesure la sensibilité du prix d'un titre aux petites variations de taux.",
            "La convexité corrige l'erreur de linéarisation dès que le mouvement de taux devient moins infinitésimal.",
            "Ces notions sont partout : obligations, books de taux, immunisation.",
        ],
        formulas=[
            "$\\frac{\\Delta P}{P} \\approx -D_{mod} \\Delta y$.",
            "$\\frac{\\Delta P}{P} \\approx -D_{mod} \\Delta y + \\tfrac{1}{2} C (\\Delta y)^2$.",
            "$D_{mod} = \\frac{D_{Mac}}{1+y/m}$ dans le cas discret standard.",
        ],
        pitfalls=[
            "Confondre duration Macaulay et duration modifiée.",
            "Oublier que l'approximation se dégrade pour des gros shifts de courbe.",
            "Parler de duration comme si elle était identique pour tous les mouvements de courbe.",
        ],
        interview_drills=[
            "Pourquoi la convexité est-elle généralement positive pour une obligation classique ?",
            "Comment interpréter une duration de 5 ?",
            "Qu'est-ce qu'une couverture duration-neutre ne couvre pas ?",
        ],
        signals=[
            "Le candidat doit verbaliser 'pente puis courbure' et pas juste réciter les formules.",
            "Il faut relier duration et approximation locale de la fonction prix-taux.",
            "Une bonne réponse mentionne les limites d'une couverture parallèle uniquement.",
        ],
    ),
    _build_sheet(
        slug="var_expected_shortfall",
        title="VaR et Expected Shortfall",
        icon="R",
        category="Risque",
        level="Intermédiaire",
        duration="8 min",
        tags=("var", "expected shortfall", "tail risk"),
        summary="Le socle des discussions risque : quantile de perte, sévérité conditionnelle en queue et limites pratiques des métriques réglementaires.",
        intuition=[
            "La VaR à 99% donne un seuil de perte rarement dépassé sur l'horizon retenu.",
            "L'Expected Shortfall regarde la gravité moyenne des pertes une fois ce seuil franchi.",
            "En pratique, l'ES est souvent préférée car elle capture mieux le risque de queue.",
        ],
        formulas=[
            "$\\text{VaR}_{\\alpha}(L)$ = plus petit seuil tel que $\\mathbb{P}(L \\le x) \\ge \\alpha$.",
            "$\\text{ES}_{\\alpha}(L) = \\mathbb{E}[L \\mid L \\ge \\text{VaR}_{\\alpha}(L)]$ sous une convention de pertes adaptée.",
            "Sous normalité, les deux métriques ont des formes fermées mais cette hypothèse peut être trompeuse.",
        ],
        pitfalls=[
            "Dire que la VaR est la perte maximale.",
            "Oublier l'horizon temporel et le niveau de confiance.",
            "Comparer des VaR entre portefeuilles sans homogénéiser les hypothèses de modélisation.",
        ],
        interview_drills=[
            "Pourquoi l'ES est-elle cohérente au sens d'Artzner alors que la VaR peut ne pas l'être ?",
            "Quelles limites de la VaR sous hypothèse gaussienne ?",
            "Comment backtester une mesure de risque de queue ?",
        ],
        signals=[
            "Le candidat doit distinguer seuil de quantile et perte conditionnelle moyenne.",
            "Une bonne réponse évoque la non-subadditivité potentielle de la VaR.",
            "Le niveau pro inclut une discussion sur le modèle, pas seulement sur la formule.",
        ],
    ),
    _build_sheet(
        slug="sabr_smile",
        title="SABR et smile de volatilité",
        icon="V",
        category="Volatilité",
        level="Élevé",
        duration="11 min",
        tags=("sabr", "smile", "stochastic volatility"),
        summary="Un classique des desks taux et FX : comment générer un smile plausible, lire les paramètres et comprendre les limites du modèle.",
        intuition=[
            "SABR fait évoluer le forward avec une volatilité elle-même stochastique et corrélée au sous-jacent.",
            "Le paramètre $\\rho$ pilote principalement l'asymétrie, et $\\nu$ la variabilité de la vol.",
            "Le modèle est apprécié pour sa lecture de smile rapide, même si sa calibration a des angles morts.",
        ],
        formulas=[
            "$dF_t = \\alpha_t F_t^{\\beta} dW_t^1$.",
            "$d\\alpha_t = \\nu \\alpha_t dW_t^2$ avec $dW_t^1 dW_t^2 = \\rho dt$.",
            "La formule implicite de Hagan relie $(\\alpha, \\beta, \\rho, \\nu)$ à la volatilité Black observée.",
        ],
        pitfalls=[
            "Présenter $\\beta$ comme un simple paramètre numérique sans interprétation.",
            "Oublier que SABR est surtout utilisé via une approximation de vol implicite.",
            "Dire que le modèle calibre parfaitement tous les smiles sans arbitrage.",
        ],
        interview_drills=[
            "Quel rôle joue $\\rho$ sur le skew ?",
            "Pourquoi fixe-t-on souvent $\\beta$ en pratique ?",
            "Quelles différences d'usage entre local vol et SABR ?",
        ],
        signals=[
            "Le candidat doit savoir raconter l'effet qualitatif de chaque paramètre.",
            "Une bonne réponse parle calibration, stabilité et interprétation de marché.",
            "Le niveau élevé se voit dans la discussion sur les limites hors calibration locale.",
        ],
    ),
    _build_sheet(
        slug="credit_intensity_models",
        title="Crédit et modèles d'intensité",
        icon="C",
        category="Crédit",
        level="Intermédiaire",
        duration="10 min",
        tags=("default", "hazard rate", "credit"),
        summary="Base utile pour CDS et crédit quant : la probabilité de défaut se modélise par une intensité, et le spread porte une information de survie sous hypothèses.",
        intuition=[
            "Dans les modèles réduits, le défaut arrive comme un saut gouverné par une intensité.",
            "La survival probability décroît avec l'intensité cumulée.",
            "Le spread de crédit reflète de façon simplifiée le risque de défaut et le recouvrement.",
        ],
        formulas=[
            "$\\mathbb{P}(\\tau > t) = e^{-\\int_0^t \\lambda_s ds}$ dans le cas général.",
            "Avec intensité constante : $\\mathbb{P}(\\tau > t) = e^{-\\lambda t}$.",
            "Approximation simple : spread $\\approx \\lambda (1-R)$ sous hypothèses stylisées.",
        ],
        pitfalls=[
            "Confondre modèle structurel et modèle d'intensité.",
            "Parler du spread comme d'une probabilité de défaut brute.",
            "Oublier l'effet du recovery dans l'interprétation du spread.",
        ],
        interview_drills=[
            "Quel lien entre hazard rate et survival curve ?",
            "Pourquoi un spread ne suffit-il pas toujours à identifier séparément défaut et recouvrement ?",
            "Comment pricer grossièrement un CDS dans un cadre simple ?",
        ],
        signals=[
            "Le candidat doit distinguer intensité instantanée et probabilité cumulée.",
            "Une bonne réponse relie spread, recovery et survie sans surpromettre la formule simple.",
            "Le niveau solide mentionne les limites des hypothèses de constance de l'intensité.",
        ],
    ),
)


COURSE_TRACKS: tuple[CourseTrack, ...] = (
    CourseTrack(
        slug="quant_core",
        title="Core quant",
        description="Le socle à maîtriser avant un entretien généraliste : diffusion, pricing et sens des hypothèses.",
        topics=("brownian_motion", "ito_lemma", "risk_neutral_pricing", "binomial_trees", "black_scholes"),
    ),
    CourseTrack(
        slug="volatility_focus",
        title="Volatilité et smile",
        description="Un parcours ciblé pour les questions de desk derivs, market making et modélisation du smile.",
        topics=("greeks", "implied_volatility", "local_vs_stochastic_vol", "sabr_smile", "monte_carlo"),
    ),
    CourseTrack(
        slug="rates_credit",
        title="Rates / credit",
        description="Pour élargir la préparation aux questions fixed income, swaps, courbes et risque de défaut.",
        topics=("interest_rate_curves", "swaps_and_swap_rate", "duration_convexity", "credit_intensity_models"),
    ),
)


def get_course_catalog() -> list[CourseSheet]:
    return list(COURSE_CATALOG)


def get_course_categories() -> list[str]:
    return sorted({sheet.category for sheet in COURSE_CATALOG})


def get_course_levels() -> list[str]:
    return sorted({sheet.level for sheet in COURSE_CATALOG}, key=lambda level: ["Fondamental", "Intermédiaire", "Élevé"].index(level))


def get_course_tracks() -> list[CourseTrack]:
    return list(COURSE_TRACKS)
