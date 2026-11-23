# Notes: why input drift and accuracy can separate, and what a label-free monitor can and cannot see

These are short notes behind the benchmark, not new theory. Both constructions below are standard
textbook examples.

## Setup

A model $f$ is trained on a source distribution $P_S(X, Y)$ and deployed on a target $P_T(X, Y)$, with
binary $Y$. Accuracy on a distribution $P$ is $\mathrm{Acc}_P(f) = P(f(X) = Y)$. The deployment is
*degraded* if $\mathrm{Acc}_T(f) < \mathrm{Acc}_S(f) - \epsilon$ (the benchmark uses $\epsilon = 0.02$).

Write the joint as $P(X, Y) = P(X)\,P(Y \mid X)$. Shift can come from either factor:

- **covariate shift**: $P_T(X) \ne P_S(X)$, $P_T(Y \mid X) = P_S(Y \mid X)$
- **concept shift**: $P_T(Y \mid X) \ne P_S(Y \mid X)$

An input-drift monitor sees only samples of $X$ from the source and target. So does every other
*label-free* monitor, including confidence-based ones, since the confidence $c(x) = \max_y \hat p(y \mid x)$
is a fixed function of $x$.

## 1. Drift without degradation

Let the source put most of its mass near the decision boundary and the target put its mass far from it,
with the same $P(Y \mid X)$. The target marginal is far from the source (any divergence is large), but
points far from the boundary are easier, so accuracy can go **up**.

The benchmark shows this happening: many public-coverage deployments drift past the PSI threshold and are
more accurate than California.

## 2. Degradation without drift

Keep $P_T(X) = P_S(X)$, so every drift score is zero in expectation. Say the model predicts 1 on a region
$R_+$ holding half the mass and 0 on the rest. If $P_S(Y = 1 \mid R_+) = 0.9$ and
$P_S(Y = 1 \mid R_-) = 0.1$, source accuracy is 0.9. Change only the labels: $P_T(Y = 1 \mid R_+) = 0.6$ and
$P_T(Y = 1 \mid R_-) = 0.4$. Target accuracy is 0.6, a 30-point drop with zero input drift.

**This example breaks confidence-based monitors too.** The model's confidences depend only on $x$, and the
distribution of $x$ did not change, so mean confidence and ATC report no change either. No method that sees
only unlabeled inputs can tell this target from the source. Garg et al. (2022) make the same point
formally: estimating target accuracy from unlabeled data is impossible without assumptions on the shift.
Label-free monitoring is therefore a cheap early signal between label refreshes, not a replacement for them.

## 3. When confidence can work

Suppose the shift is covariate shift and the model's confidence is right pointwise:
$P(f(X) = Y \mid X = x) = c(x)$ for every $x$. Then for any target with the same $P(Y \mid X)$,

$$\mathbb{E}_T[c(X)] = \mathbb{E}_T\big[P(f(X) = Y \mid X)\big] = \mathrm{Acc}_T(f),$$

so mean confidence is unbiased. Ordinary (binned) calibration on the source is weaker than this and does
not carry over to a new $P(X)$, which is one reason raw mean confidence can be off.

ATC (Garg et al., ICLR 2022) does not use confidence as a probability. It picks a threshold $t$ on labeled
source data so that the share of source points with $c(x) > t$ equals source accuracy, then reports the share
of target points above $t$. It is exact on the source by construction, and it stays close when the shift
mostly moves points across confidence levels without changing how errors relate to confidence.

In the benchmark, the model's mean confidence drops less than its accuracy does (it is under-confident on
the shifted states), which is why the raw confidence alarm catches 20 of 44 degraded deployments and ATC
catches 35.

## 4. Relation to the domain adaptation bound

Ben-David et al. (2010) bound target error by

$$\epsilon_T(h) \le \epsilon_S(h) + \tfrac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T) + \lambda,$$

where $d_{\mathcal{H}\Delta\mathcal{H}}$ is a divergence between the input marginals and $\lambda$ is the
error of the best single hypothesis on both domains. Construction 2 has zero marginal divergence, and the
drop shows up through $\lambda$, which depends on labels. Construction 1 shows the bound can be loose: a
large divergence does not force a large error.

## What this means for the benchmark

The cross-state ACS shifts are a mix of both kinds. The benchmark measures how the signals behave on those
real shifts. It does not prove that ATC works in general, and the constructions above show it cannot.

## References

- S. Garg, S. Balakrishnan, Z. Lipton, B. Neyshabur, H. Sedghi. *Leveraging Unlabeled Data to Predict
  Out-of-Distribution Performance.* ICLR 2022.
- S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, J. Vaughan. *A theory of learning from
  different domains.* Machine Learning, 2010.
- F. Ding, M. Hardt, J. Miller, L. Schmidt. *Retiring Adult: New Datasets for Fair Machine Learning.*
  NeurIPS 2021.
