# Notes: why input drift and accuracy degradation can separate

## 3.1 Formal Setup

Let $\mathcal{X} \subseteq \mathbb{R}^d$ be an input space and $\mathcal{Y} = \{0, 1\}$ the label space.

**Definition 1 (Training Distribution):** $P_{\text{train}}(X, Y)$ is the joint distribution over input-label pairs used to train model $f: \mathcal{X} \rightarrow \mathcal{Y}$.

**Definition 2 (Deployment Distribution):** $P_{\text{test}}(X, Y)$ is the joint distribution encountered during deployment.

**Definition 3 (Covariate Shift):** Covariate shift occurs when $P_{\text{test}}(X) \neq P_{\text{train}}(X)$ but $P_{\text{test}}(Y|X) = P_{\text{train}}(Y|X)$.

**Definition 4 (Model Degradation):** Degradation of magnitude $\epsilon$ occurs when:
$$\text{Acc}(f, P_{\text{test}}) < \text{Acc}(f, P_{\text{train}}) - \epsilon$$

**Definition 5 (Drift Detector):** A drift detector $D: \{X_1,...,X_n\} \rightarrow \{0,1\}$ maps a sample to a binary decision based solely on input statistics.

**Definition 6 (Silent Degradation):** Degradation that occurs without detectable covariate shift:
- $\text{Acc}(f, P_{\text{test}}) < \text{Acc}(f, P_{\text{train}}) - \epsilon$ (degradation present)
- $D(X_{\text{test}}) = 0$ for any drift detector $D$ (no drift detected)

---

## 3.2 The Separation Theorem

**Theorem 1 (Separation of Drift and Degradation):**

For any model $f$ and any $\epsilon > 0$, there exist distributions $P_{\text{test}}$ such that:

**(a) Undetectable Degradation:**
$$d(P_{\text{test}}(X), P_{\text{train}}(X)) < \delta \text{ for arbitrarily small } \delta > 0$$
yet
$$\text{Acc}(f, P_{\text{test}}) < \text{Acc}(f, P_{\text{train}}) - \epsilon$$

**(b) Detectable Non-Degradation:**
$$d(P_{\text{test}}(X), P_{\text{train}}(X)) > \tau \text{ for any threshold } \tau$$
yet
$$\text{Acc}(f, P_{\text{test}}) \geq \text{Acc}(f, P_{\text{train}})$$

where $d(\cdot,\cdot)$ is any divergence measure on marginal distributions.

### Proof

**Part (a) - Constructive Existence:**

Consider a binary classification task with decision boundary $\mathcal{B}$ partitioning $\mathcal{X}$ into regions $R_+$ (predicted positive) and $R_-$ (predicted negative).

Let $P_{\text{train}}$ be a distribution where:
- Support is uniform over $\mathcal{X}$
- $P(Y=1|X \in R_+) = 0.9$, $P(Y=1|X \in R_-) = 0.1$

Define $P_{\text{test}}$ with identical marginal $P_{\text{test}}(X) = P_{\text{train}}(X)$, but:
- $P(Y=1|X \in R_+) = 0.6$, $P(Y=1|X \in R_-) = 0.4$

The marginals are **identical**, so $d(P_{\text{test}}(X), P_{\text{train}}(X)) = 0$.

However:
- Train accuracy: $0.5 \cdot 0.9 + 0.5 \cdot 0.9 = 0.9$ (90%)
- Test accuracy: $0.5 \cdot 0.6 + 0.5 \cdot 0.6 = 0.6$ (60%)
- **Degradation: 30%** (massive!)

This constructs a distribution with **ZERO drift** but **MASSIVE degradation**. ∎

**Part (b) - Benign Shift:**

Let $P_{\text{train}}$ be concentrated near the decision boundary $\mathcal{B}$.
Let $P_{\text{test}}$ be shifted far from $\mathcal{B}$ but with the same class-conditional structure.

Then:
- $d(P_{\text{test}}(X), P_{\text{train}}(X))$ is large (different regions of $\mathcal{X}$)
- But accuracy may **improve** (farther from boundary = easier classification)

This constructs a distribution with **LARGE drift** but **NO degradation**. ∎

---

## 3.3 Information-Theoretic Lower Bound

**Theorem 2 (No-Free-Lunch for Drift Detection):**

Let $D$ be any drift detector that observes only inputs $\{X_1,...,X_n\}$.
Let $\Delta(f, P_{\text{test}}) = \text{Acc}(f, P_{\text{train}}) - \text{Acc}(f, P_{\text{test}})$ denote degradation.

Then:
$$I(D(X); \Delta(f, P_{\text{test}})) \leq I(P_{\text{test}}(X); P_{\text{test}}(Y|X))$$

where $I(\cdot;\cdot)$ denotes mutual information.

**Interpretation:** The information a drift detector can provide about degradation is bounded by the information the marginal distribution provides about the conditional distribution. Under concept drift (where $P(Y|X)$ changes), this bound is typically **zero**.

### Proof Sketch

By the data processing inequality:
$$I(D(X); \Delta) \leq I(X; \Delta) = I(X; g(Y, \hat{Y}))$$

where $g$ computes accuracy from true and predicted labels.

Since $D$ only observes $X$, not $Y$, and $\hat{Y} = f(X)$ is deterministic:
$$I(D(X); \Delta) \leq H(\Delta) - H(\Delta | X)$$

But $\Delta$ depends on $P(Y|X)$, which is **not observable** from $X$ alone. Thus the conditional entropy $H(\Delta | X)$ remains high, and the mutual information approaches zero. ∎

---

## 3.4 Connection to Established Theory

### Relationship to Ben-David et al. (2010)

The classic domain adaptation bound states:
$$\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(D_S, D_T) + \lambda$$

where $\lambda$ is the optimal joint error. Our result shows that even when $d_{\mathcal{H}\Delta\mathcal{H}} \approx 0$ (no distributional divergence), $\lambda$ can be arbitrarily large, causing degradation.

### Relationship to Importance Weighting

Under covariate shift, importance weighting corrects for $P_{\text{test}}(X) / P_{\text{train}}(X)$. Our result shows this is **insufficient** when $P(Y|X)$ changes—the weights are correct but the labels are wrong.

### Relationship to Conformal Prediction

Conformal prediction provides valid coverage under exchangeability. Our work shows that coverage validity does not imply accuracy preservation—a model can have valid confidence sets but still be wrong.

---

## 3.5 Implications

**Implication 1:** No amount of input monitoring can guarantee detection of all degradation.

**Implication 2:** Effective monitoring must incorporate information **downstream** of the decision boundary—i.e., model outputs or predictions.

**Implication 3:** Confidence distribution monitoring (our proposed alternative) satisfies this requirement by observing $P(f(X))$ rather than just $P(X)$.

---

## 3.6 Why Confidence Monitoring Works

**Proposition 1:** If model $f$ is calibrated, then:
$$\mathbb{E}[\max_y P(Y=y|X)] = \text{Acc}(f)$$

Thus, monitoring the confidence distribution directly tracks accuracy.

**Proposition 2:** Even under miscalibration, the **relative change** in confidence distribution correlates with accuracy change:
$$\text{Corr}(\Delta\text{Conf}, \Delta\text{Acc}) > 0$$

This explains our empirical finding of $r > 0.87$ across all benchmarks.
