# KFAC-Muon Paper Drafting Handoff

This document is intended to be pasted into a longer-context / higher-reasoning ChatGPT session to help draft a machine-learning paper from the current KFAC-Muon project state. It combines project context, current write-up status, derivation goals, implementation details, experiment history, exact known results, caveats, and suggested next steps.

## Request For The Next Model

You are helping draft a machine-learning research paper about **KFAC-Muon**, an optimizer that combines Muon-style orthogonalized matrix updates with KFAC/Fisher whitening. The user already has a partial LaTeX write-up and preliminary experiments. Your job is to turn the current material into a coherent paper draft while being careful not to overclaim beyond the available results.

Please do the following:

1. Use the current project state below as ground truth.
2. Preserve the core thesis: KFAC-Muon applies Muon in KFAC-whitened coordinates, producing an orthogonal update constrained in approximate Fisher geometry.
3. Fill in missing derivation sections carefully and consistently.
4. Clean up the paper structure and remove duplicated placeholder sections.
5. Strengthen the experiment section using the exact results below.
6. Clearly mark preliminary/single-seed results as such unless additional seed results are provided later.
7. Do not invent experiments, numbers, or citations.
8. If a claim depends on a missing experiment, either soften the claim or list it as future work.

The user wants a paper draft, not just notes. The style should be clear, technically precise, and ML-paper-like, but it can remain a working draft.

---

# Project Summary

## Project Name

KFAC-Muon

## One-Sentence Thesis

**KFAC-Muon improves Muon by applying Muon orthogonalization in KFAC-whitened coordinates, so the spectral-norm-bounded update is taken in an approximate Fisher/natural-gradient geometry rather than raw parameter space.**

## Core Idea

Muon is an optimizer for matrix-valued parameters. For a matrix parameter `W`, Muon builds a momentum matrix `M`, approximately computes its polar factor using Newton-Schulz iterations, and updates:

```math
W_{t+1} = W_t - \eta_t \operatorname{Polar}(M_t).
```

Muon can be interpreted as solving a linear minimization problem under a spectral-norm constraint:

```math
\eta_t \operatorname{Polar}(M_t)
\in
\arg\min_{\|S\|_2 \le \eta_t} -\langle M_t, S\rangle_F.
```

KFAC approximates a layerwise Fisher block for an affine map `s = W a` using Kronecker factors:

```math
F_W \approx A \otimes \Gamma,
\qquad
A = \mathbb{E}[a a^\top],
\qquad
\Gamma = \mathbb{E}[\delta \delta^\top],
```

where `a` is the layer input/activation and `delta = \nabla_s \ell` is the backpropagated output gradient.

KFAC-Muon combines these ideas:

1. Maintain Muon-style momentum `M_t` in parameter coordinates.
2. Compute KFAC factors `A_t`, `Gamma_t`.
3. Form damped Cholesky factors:

```math
A_\lambda = A + \lambda_A I,\qquad
\Gamma_\lambda = \Gamma + \lambda_\Gamma I.
```

4. Whiten momentum:

```math
\widehat M_t = L_\Gamma^{-1} M_t L_A^{-\top}.
```

5. Apply Muon/Newton-Schulz in whitened coordinates:

```math
\widehat O_t = \operatorname{NewtonSchulz}(\widehat M_t).
```

6. Unwhiten:

```math
O_t = L_\Gamma^{-\top} \widehat O_t L_A^{-1}.
```

7. Update:

```math
W_{t+1} = W_t - \eta_t O_t.
```

The intended interpretation is: Muon normally constrains the spectral norm of the step in Euclidean parameter geometry. KFAC-Muon constrains the spectral norm of the step after transformation into Fisher/KFAC-whitened geometry, making the update more conservative in directions that are sensitive under the model's predictive distribution.

---

# Current Draft State

The current PDF is about 11 pages and contains:

1. Introduction
2. Background
   - Muon as a spectral-norm update
   - Natural gradient and KFAC
3. KFAC-Muon
   - Algorithm equation exists
   - Several subsections are placeholders
4. Experiments
   - Setup
   - Main results
   - FISMO-style comparison placeholder
   - Ablations
   - Runtime and memory placeholder
   - Diagnostics
5. Related Work and Discussion placeholder
6. Then duplicated scaffold sections appear again:
   - Notation and Background
   - Fisher-Whitened Orthogonal Updates
   - KFAC-Muon
   - Experiments
   - Related Work and Conclusion
   - Appendices

## Important Structural Issue

The current PDF still has duplicate placeholder sections after the main experiment section. The next draft should remove/merge the duplicated scaffold and settle on a clean structure.

Suggested clean structure:

```latex
\section{Introduction}
\section{Background}
  \subsection{Muon as a Spectral-Norm Update}
  \subsection{Natural Gradient and KFAC}
\section{Fisher-Whitened Orthogonal Updates}
  \subsection{A Kronecker-Metric Spectral Trust Region}
  \subsection{Closed-Form Whiten--Polar--Unwhiten Update}
  \subsection{Relation to FISMO}
\section{KFAC-Muon}
  \subsection{KFAC Factors for Affine Layers}
  \subsection{KFAC-Reduce for Shared-Weight Layers}
  \subsection{Algorithm}
  \subsection{Practical Implementation Details}
\section{Experiments}
  \subsection{Setup}
  \subsection{Main Results}
  \subsection{Ablations}
  \subsection{Runtime and Memory}
  \subsection{Diagnostics}
\section{Related Work}
\section{Conclusion}
\appendix
  \section{Derivations}
  \section{Exact Shared-Weight Fisher and KFAC-Reduce}
  \section{Additional Experiments and Hyperparameters}
```

The current draft has useful material in Introduction, Background, and Experiments. The biggest missing writing work is the derivation and shared-weight/KFAC-reduce section.

---

# Current Contributions To Claim

The intro currently lists contributions along these lines:

1. Derive KFAC-Muon as Muon applied in KFAC-whitened coordinates, giving an orthogonal update constrained in approximate Fisher geometry.
2. Develop a KFAC-reduce approximation for shared-weight affine layers, allowing KFAC-Muon to be applied efficiently to transformer-style architectures.
3. Compare KFAC-Muon to Muon and FISMO-style fixed-point variants, showing that the Fisher/Kronecker estimator improves performance.
4. Empirically show KFAC-Muon improves over standard Muon on image-classification tasks.

## Important Caveat About Contribution 3

The controlled comparison to FISMO-style factor estimation is not completed yet for the final recipes. The draft should either:

- soften this contribution to say the paper is related to FISMO and motivates a future comparison, or
- leave a clear TODO for the experiment, or
- include it only if the user later provides results.

A safer contribution list right now:

1. We derive KFAC-Muon as a whiten--polar--unwhiten update: Muon applied under a Kronecker-factored approximation to Fisher geometry.
2. We describe KFAC-reduce, a practical estimator for shared-weight affine layers such as ViT patch embeddings and transformer linear projections.
3. We implement KFAC-Muon for ViT-style image classifiers, including practical damping, factor update, and layer-coverage choices.
4. In preliminary single-seed experiments on CIFAR-100 and ImageNet-100, KFAC-Muon improves over tuned Muon baselines.

---

# Theory / Derivation To Write

## Muon Variational View

Muon for a matrix momentum `M` approximately computes `Polar(M)`. If `M = U Sigma V^T`, then:

```math
\operatorname{Polar}(M) = U V^\top.
```

It solves:

```math
\operatorname{Polar}(M)
\in
\arg\max_{\|O\|_2 \le 1} \langle M, O\rangle_F.
```

Equivalently, the step `S = \eta O` solves:

```math
S^\star
\in
\arg\min_{\|S\|_2 \le \eta}
-\langle M, S\rangle_F.
```

This should be stated cleanly using von Neumann's trace inequality or duality between spectral and nuclear norms.

## KFAC-Metric Spectral Trust Region

For affine layer `W`, KFAC approximates the Fisher block as:

```math
F_W \approx A \otimes \Gamma.
```

The Fisher quadratic of a step `S` is:

```math
\operatorname{vec}(S)^\top (A \otimes \Gamma) \operatorname{vec}(S)
= \operatorname{tr}(S^\top \Gamma S A)
= \|\Gamma^{1/2} S A^{1/2}\|_F^2.
```

KFAC-Muon is not simply constraining this Frobenius/Fisher norm. It uses a **spectral** constraint in whitened coordinates:

```math
\|\Gamma^{1/2} S A^{1/2}\|_2 \le \eta.
```

Or, depending on convention with Cholesky factors:

```math
\widehat S = L_\Gamma^\top S L_A,
\qquad
\|\widehat S\|_2 \le \eta.
```

Need to ensure consistency with the algorithm's whitening/unwhitening convention. The current algorithm uses:

```math
\widehat M = L_\Gamma^{-1} M L_A^{-\top},
\widehat O = \operatorname{Polar}(\widehat M),
O = L_\Gamma^{-\top} \widehat O L_A^{-1}.
```

This corresponds to defining the whitened step as:

```math
\widehat S = L_\Gamma^\top S L_A.
```

Then:

```math
S = L_\Gamma^{-\top} \widehat S L_A^{-1}.
```

The objective transforms as:

```math
\langle M, S\rangle_F
= \langle M, L_\Gamma^{-\top} \widehat S L_A^{-1}\rangle_F
= \langle L_\Gamma^{-1} M L_A^{-\top}, \widehat S\rangle_F
= \langle \widehat M, \widehat S\rangle_F.
```

Thus the constrained linear problem becomes:

```math
\widehat S^\star
\in
\arg\min_{\|\widehat S\|_2 \le \eta}
-\langle \widehat M, \widehat S\rangle_F,
```

whose solution is:

```math
\widehat S^\star = \eta \operatorname{Polar}(\widehat M)
```

up to the sign convention for descent. Then:

```math
S^\star = \eta L_\Gamma^{-\top}\operatorname{Polar}(L_\Gamma^{-1} M L_A^{-\top})L_A^{-1}.
```

This is the core derivation.

## Relation To Natural Gradient

Natural gradient solves a linearized objective under a Fisher/Frobenius trust region:

```math
\min_S \langle G, S\rangle_F
\quad\text{s.t.}\quad
\operatorname{vec}(S)^\top F \operatorname{vec}(S) \le \rho^2.
```

Under KFAC, this gives:

```math
S \propto -\Gamma^{-1} G A^{-1}.
```

KFAC-Muon replaces the Fisher-Frobenius ball with a Fisher-whitened spectral-norm ball. This gives an orthogonalized update in the whitened geometry rather than a purely natural-gradient direction.

Possible phrasing:

> Natural gradient controls the Frobenius norm of the whitened step; Muon controls the spectral norm of the raw step. KFAC-Muon controls the spectral norm of the KFAC-whitened step.

This is a nice conceptual triangle.

## KFAC Factors For Affine Layers

For `s = W a`, per-example gradient:

```math
\nabla_W \ell = \delta a^\top.
```

Vectorized outer product:

```math
\operatorname{vec}(\delta a^\top)\operatorname{vec}(\delta a^\top)^\top
= (a a^\top) \otimes (\delta \delta^\top).
```

KFAC approximation:

```math
\mathbb{E}[(a a^\top)\otimes(\delta\delta^\top)]
\approx
\mathbb{E}[a a^\top] \otimes \mathbb{E}[\delta\delta^\top].
```

## KFAC-Reduce For Shared-Weight Layers

This section is important and currently incomplete.

For transformers/ViTs, a matrix `W` is applied at many token/spatial positions. For example, for token position `t` in example `b`:

```math
s_{b,t} = W x_{b,t}.
```

The per-example gradient sums over positions:

```math
G_b = \sum_t \delta_{b,t} x_{b,t}^\top.
```

The exact empirical Fisher term is:

```math
\operatorname{vec}(G_b)\operatorname{vec}(G_b)^\top
= \sum_{t,u}
(x_{b,t} x_{b,u}^\top) \otimes (\delta_{b,t}\delta_{b,u}^\top).
```

This has token-token cross terms and is not simply the product of per-token activation and error covariances.

The implemented KFAC-reduce approximation uses a reduced activation and reduced error per example. From the code/history:

- For linear/shared affine layers, it reduces activations across tokens/positions and sums output gradients.
- For Conv2d/patch embedding, it unfolds patches, averages activations over spatial positions, sums gradients over spatial positions, then forms KFAC factors.

For Conv2d specifically, code behavior:

- Use `F.unfold` to collect convolutional patches.
- Average activation patches across spatial positions:

```math
\bar a_b = \frac{1}{T}\sum_t a_{b,t}.
```

- Sum output gradients across spatial positions:

```math
\bar\delta_b = \sum_t \delta_{b,t}.
```

- Form:

```math
A_{red} = \frac{1}{B}\sum_b \bar a_b \bar a_b^\top,
\qquad
\Gamma_{red} = \frac{1}{B}\sum_b \bar\delta_b \bar\delta_b^\top.
```

Need to verify exact scaling for linear layers in code if writing a formal equation. For the paper, it is okay to describe it as a practical reduce approximation and place exact scaling in appendix.

Potential language:

> KFAC-reduce treats the shared matrix as producing a single reduced per-example gradient by aggregating over the sharing dimension before forming Kronecker factors. This preserves the batch dimension needed for empirical Fisher statistics while avoiding token-token Fisher blocks.

## Patch Embedding In ViT

The ViT patch embedding is implemented as a Conv2d with kernel/stride equal to patch size. For ViT-B/16:

- Patch embedding Conv2d weight shape is roughly `[768, 3, 16, 16]`.
- It can be reshaped as a matrix `[768, 768]`.
- KFAC treats this as an affine layer over unfolded image patches.

This turned out to matter empirically. Including the patch embedding and classifier head was important.

---

# Implementation Details

Current implementation is in `train.py` in the repo `/Users/luke/kfac muon`.

Important implementation facts:

## KFAC-Muon Parameter Selection

The code identifies candidate affine modules:

- `nn.Linear`
- `nn.Conv2d` with `groups == 1`

Earlier default behavior excluded the first and last affine layers from KFAC-Muon. This hurt ViT performance. The successful runs include first/last layers using:

```bash
--no-kfac-exclude-first-last
```

In code, relevant parser flags:

```python
--kfac-exclude-first-last
--no-kfac-exclude-first-last
```

Current parser defaults in the local code appear to set:

```python
kfac_exclude_first_last=False
```

but older runs/scripts may differ, so experiment args should always state this explicitly.

## KFAC-Muon Step

For each KFAC module:

1. Collect activation and output-gradient statistics.
2. Maintain EMA of factors.
3. Dampen factors.
4. Cholesky or inverse-factor precondition in/out.
5. Apply Newton-Schulz Muon orthogonalization in whitened coordinates.
6. Unwhiten and apply update.

## Fast KFAC Path

A major runtime improvement was using inverse factors rather than triangular solves:

```bash
--kfac-use-inverse-factors
--kfac-inverse-compute-dtype bfloat16
```

For ImageNet-100 final run, this was used.

For the best CIFAR include-first/last run, exact `args.yaml` was not recovered locally. It likely did **not** use inverse factors, based on the older CIFAR script lineage, but this should be verified if possible.

## Muon Reference Diagnostics

KFAC summaries sometimes include:

- `train_kfac_eff_step_ratio`
- `train_kfac_step_norm`
- `train_kfac_grad_norm`
- `train_kfac_step_rms`
- `train_kfac_grad_rms`
- `train_kfac_grad_step_cos`
- `train_kfac_vs_muon_step_ratio`
- `train_kfac_muon_ref_step_norm`
- `train_kfac_muon_ref_step_rms`

`kfac_vs_muon_step_ratio` compares KFAC-Muon step size to a plain Muon reference step. It is useful for tuning but not necessary for the main paper unless used in diagnostics.

## Damping

Important distinction:

- Some runs used adaptive Levenberg-Marquardt damping.
- Final/best ImageNet-100 run used static damping `5e-5`.
- Best CIFAR include-first/last run also used static damping `5e-5`.

Damping too small increased step sizes but did not reliably improve validation performance. Damping around `5e-5` was good for the final ViT runs.

---

# Experiment Results

## Main Single-Seed Results

These are current best known paper results.

| Dataset / Model | Optimizer | Epochs | Best top-1 | Final top-1 | Final loss |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 / ViT-S/16 | Muon | 75 | 73.53 | 73.53 | 1.167 |
| CIFAR-100 / ViT-S/16 | KFAC-Muon | 75 | 75.70 | 75.70 | 1.084 |
| ImageNet-100 / ViT-B/16 | Muon | 90 | 82.30 | 82.30 | 0.854 |
| ImageNet-100 / ViT-B/16 | KFAC-Muon | 90 | 84.06 | 84.06 | 0.807 |

Gains:

- CIFAR-100 ViT-S/16: `+2.17` top-1 points.
- ImageNet-100 ViT-B/16: `+1.76` top-1 points.

Both are single-seed results.

## CIFAR-100 Details

### Baseline Muon Run

Run name:

```text
vits16_c100_muon_lr1p2e3_e75_s12
```

Local summary:

```text
/Users/luke/Downloads/vits16_c100_last_sweep/summaries/vits16_c100_muon_lr1p2e3_e75_s12/summary.csv
```

Args:

```text
/Users/luke/Downloads/vits16_c100_last_sweep/args/vits16_c100_muon_lr1p2e3_e75_s12/args.yaml
```

Key config:

```text
model: vit_small_patch16_224
num_classes: 100
dataset: CIFAR-100 ImageFolder
epochs: 75
seed: 12
batch_size: 128
validation_batch_size: 128
lr: 0.0012
weight_decay: 0.07
warmup_epochs: 10
min_lr: 1e-5
mixup: 0.2
cutmix: 0.2
mixup_off_epoch: 56
smoothing: 0.1
reprob: 0.1
drop_path: 0.1
amp_dtype: bfloat16
opt: muon
```

Result:

```text
best top-1: 73.53 at epoch 74
final top-1: 73.53
final loss: 1.16668
```

### Best KFAC-Muon CIFAR Run

Run name:

```text
vits16_c100_kfac_lmoff_damp5e5_inclfl_e75_s12
```

Local summary:

```text
/Users/luke/Downloads/vits16_c100_partial/summaries/vits16_c100_kfac_lmoff_damp5e5_inclfl_e75_s12/summary.csv
```

Important caveat: exact `args.yaml` is not local. It should be downloaded from the original Vast machine if possible:

```text
/workspace/logs/timm_train/vits16_c100_kfac_lmoff_damp5e5_inclfl_e75_s12/args.yaml
```

Known/reconstructed config:

```text
model: vit_small_patch16_224
num_classes: 100
dataset: CIFAR-100 ImageFolder
epochs: 75
seed: 12
batch_size: 128
validation_batch_size: 128
lr: 5.5e-4
weight_decay: 0.07
warmup_epochs: 8
min_lr: 1e-5
mixup: 0.2
cutmix: 0.2
mixup_off_epoch: 56
smoothing: 0.1
reprob: 0.1
drop_path: 0.1
amp_dtype: bfloat16
opt: kfac_muon
kfac_damping: 5e-5
kfac_lm_adapt_damping: false
kfac_muon_eps: 0.038
kfac_muon_lr_adjustment: match_rms_adamw
kfac_momentum: 0.9
kfac_stats_update_every: 2
kfac_factor_update_every: 2
kfac_ema_decay: 0.95
kfac_track_muon_reference: true
kfac_exclude_first_last: false
```

Key distinguishing flags likely:

```bash
--no-kfac-lm-adapt-damping \
--kfac-damping 5e-5 \
--no-kfac-exclude-first-last
```

Result:

```text
best top-1: 75.70 at epoch 74
final top-1: 75.70
final loss: 1.08448
best loss: 1.07954 at epoch 72
```

### Older CIFAR KFAC Run Used Before Updating Draft

Run name:

```text
vits16_c100_kfac_rho_0p9_2p1_e75_s12
```

Result:

```text
best/final top-1: 75.05
final loss: 1.091
```

This is now superseded by the include-first/last static-damping run above.

### Longer CIFAR Sanity Check

Muon:

```text
/Users/luke/Downloads/kfac_runs/vits16_cifar100_muon_e200_baseline/summary.csv
best top-1: 74.83
final top-1: 74.81
```

KFAC-Muon:

```text
/Users/luke/Downloads/kfac_runs/vits16_cifar100_kfacmuon_e200_adapt_baseline/summary.csv
best top-1: 75.86
final top-1: 75.85
```

This supports the same direction, but the 75-epoch include-first/last run is currently cleaner and stronger for the main table.

## ImageNet-100 Details

Dataset is ImageNet-100, 100-class subset of ImageNet. The user used `/workspace/data/imagenet100` or `/workspace/data` depending on machine, but final commands use `/workspace/data/imagenet100`.

### Baseline Muon ImageNet-100 b256 e90

Run name:

```text
vitb16_in100_muon_e90_lr1e3_b256_seed11
```

Local summary:

```text
/Users/luke/Downloads/muon_b256_e90/summary.csv
```

Config:

```text
model: vit_base_patch16_224
num_classes: 100
epochs: 90
seed: 11
batch_size: 256
validation_batch_size: 256
lr: 1e-3
weight_decay: 0.07
warmup_epochs: 5
min_lr: 1e-5
mixup: 0.2
cutmix: 0.2
mixup_off_epoch: 70
smoothing: 0.1
reprob: 0.1
drop_path: 0.1
amp_dtype: bfloat16
opt: muon
```

Result:

```text
best top-1: 82.30 at epoch 89
final top-1: 82.30
final loss: 0.85432
```

### KFAC-Muon ImageNet-100 b256 e90

Run name:

```text
vitb16_in100_kfac_muon_e90_lr1e3_b256_d5e5_eps0p012_include_firstlast_seed11
```

Local summary:

```text
/Users/luke/Downloads/kfac_b256_e90/summary.csv
```

Config:

```text
model: vit_base_patch16_224
num_classes: 100
epochs: 90
seed: 11
batch_size: 256
validation_batch_size: 256
lr: 1e-3
weight_decay: 0.07
warmup_epochs: 5
min_lr: 1e-5
mixup: 0.2
cutmix: 0.2
mixup_off_epoch: 70
smoothing: 0.1
reprob: 0.1
drop_path: 0.1
amp_dtype: bfloat16
opt: kfac_muon
kfac_damping: 5e-5
kfac_muon_eps: 0.012
kfac_momentum: 0.9
kfac_stats_update_every: 2
kfac_factor_update_every: 2
kfac_ema_decay: 0.95
kfac_lm_adapt_damping: false
kfac_use_inverse_factors: true
kfac_inverse_compute_dtype: bfloat16
kfac_track_muon_reference: false
kfac_exclude_first_last: false
```

Important flags:

```bash
--kfac-use-inverse-factors \
--kfac-inverse-compute-dtype bfloat16 \
--kfac-ema-decay 0.95 \
--no-kfac-lm-adapt-damping \
--no-kfac-track-muon-reference \
--no-kfac-exclude-first-last
```

Result:

```text
best top-1: 84.06 at epoch 89
final top-1: 84.06
final loss: 0.80722
```

Important milestone comparison:

| Epoch | Muon top-1 | KFAC top-1 | Gap |
|---:|---:|---:|---:|
| 5 | 46.28 | 51.74 | +5.46 |
| 10 | 61.38 | 65.18 | +3.80 |
| 20 | 71.38 | 72.84 | +1.46 |
| 40 | 77.88 | 78.58 | +0.70 |
| 60 | 79.40 | 81.12 | +1.72 |
| 80 | 81.90 | 83.44 | +1.54 |
| 89 | 82.30 | 84.06 | +1.76 |

Threshold timing:

| Threshold | Muon reaches | KFAC reaches |
|---:|---:|---:|
| 78% | epoch 42 | epoch 35 |
| 80% | epoch 58 | epoch 53 |
| 82% | epoch 85 | epoch 64 |
| 83% | never | epoch 76 |
| 84% | never | epoch 89 |

Interpretation: KFAC-Muon is not merely ahead early; it remains ahead through the end and reaches high-accuracy thresholds earlier.

## Older ImageNet-100 b64 Result

There was an older Muon b64 e90 baseline:

```text
/Users/luke/Downloads/muon_baselines/vitb16_in100_muon_e90_lr1.0e-3_seed11/summary.csv
```

Result:

```text
best top-1: 84.46 at epoch 83
final top-1: 84.08
```

Do **not** compare this as the main baseline for the final ImageNet table, because it used batch size 64 and therefore 4x more optimizer steps per epoch than the b256 runs. It is useful context only.

## Batch Size Notes

The final ImageNet-100 result uses batch size 256 because it is closer to common ViT training recipes and fairer for paper-facing comparisons. Batch size 512 OOMed on available hardware. Batch size 64 was easier but less standard and had more optimizer steps per epoch.

---

# Figures and Tables

Current generated artifacts in the repo:

```text
/Users/luke/kfac muon/notes/experiments_section.tex
/Users/luke/kfac muon/notes/figures/cifar100_vits16_curves.pdf
/Users/luke/kfac muon/notes/figures/imagenet100_vitb16_b256_curves.pdf
/Users/luke/kfac muon/notes/figures/main_results_best_top1.pdf
/Users/luke/kfac muon/notes/figures/validation_loss_curves.pdf
/Users/luke/kfac muon/notes/tables/main_results.csv
/Users/luke/kfac muon/notes/make_experiment_figures.py
```

Figure descriptions:

1. `cifar100_vits16_curves.pdf`
   - Left: CIFAR-100 top-1 curves for Muon vs KFAC.
   - Right: KFAC minus Muon top-1 gap.

2. `imagenet100_vitb16_b256_curves.pdf`
   - Left: ImageNet-100 top-1 curves for Muon vs KFAC.
   - Right: KFAC minus Muon top-1 gap.

3. `main_results_best_top1.pdf`
   - Bar chart of best top-1 for CIFAR-100 and ImageNet-100.

4. `validation_loss_curves.pdf`
   - Loss curves for main CIFAR-100 and ImageNet-100 comparisons.

The figures were intentionally stripped of extra title/footer text. Captions carry explanation.

---

# Important Empirical Lessons

## 1. Including First/Last Layers Matters

This was the most important debugging discovery.

For ViTs, the patch embedding and classifier head are high-leverage affine layers. Excluding them from KFAC-Muon led to worse performance, even when KFAC step scales were adjusted. Including them substantially improved both CIFAR and ImageNet-100 results.

Paper language:

> We found that optimizer coverage is important for ViTs. Applying KFAC-Muon to the patch embedding and classifier head improved performance, suggesting that first/last affine layers are not safely treated as auxiliary parameters in this setting.

## 2. Matched Batch Size Matters

Early runs used batch size 64. Final ImageNet-100 comparison uses batch size 256. This is more defensible and avoids comparing KFAC with fewer/more optimizer steps.

## 3. Static Damping Worked Well

Adaptive damping was tested extensively but the final strongest runs use static damping `5e-5`. Adaptive damping may still be interesting, but it is not necessary for the current main result.

## 4. KFAC-Muon Gains Are Not Purely Early-Training Noise

On ImageNet-100, KFAC is ahead early and remains ahead at epoch 90. Final/best top-1 occurs at final epoch. Validation loss is consistently lower late in training.

## 5. Runtime Is A Caveat

KFAC-Muon is slower per step than Muon. Fast inverse factors with bfloat16 substantially improved overhead, but a final runtime/memory table is missing.

---

# Runtime / Profiling Context

Past profiling showed KFAC overhead was large in initial implementation. Important improvements:

- Use inverse factors rather than triangular solves.
- Compute inverse-factor matmuls in bfloat16.
- Disable Muon reference tracking for final ImageNet runs to reduce overhead.

Earlier profile examples:

- Plain Muon step around `24 ms` in one profile.
- KFAC solve-based/profile paths were much slower.
- Fast inverse factor path reduced KFAC step time substantially, though exact final b256 runtime table is not yet done.

For the paper, need a proper matched runtime table:

| Run | Hardware | Batch | Examples/sec | Step time | Epoch time | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| Muon | same GPU | 256 | TODO | TODO | TODO | TODO |
| KFAC-Muon | same GPU | 256 | TODO | TODO | TODO | TODO |

Do not overclaim runtime efficiency until this is measured.

---

# Full ImageNet-1K Next Experiment

The user asked whether to try full ImageNet. Recommendation given:

Yes, but staged.

## Why Try It

- Removes criticism that ImageNet-100 is too small/idiosyncratic.
- Tests scaling to real ImageNet-1K.
- Makes the paper much more credible.

## Caveat

ImageNet-1K has ~10x the data of ImageNet-100, so 90 epochs is ~10x compute. ViT-B/16 from scratch often uses longer recipes, e.g. DeiT-style 300 epochs. A 90-epoch full ImageNet run is a good optimizer comparison but not necessarily a best-possible ViT recipe.

## Recommended Staged Plan

1. Run a 30-epoch paired pilot on ImageNet-1K:
   - Muon
   - KFAC-Muon
   - Same seed, batch size 256, same recipe scaled to 30 epochs.

2. If KFAC is ahead, run matched 90-epoch pair.

3. Only after one successful full ImageNet seed, decide on more seeds or longer schedule.

Suggested full ImageNet KFAC recipe mirrors ImageNet-100:

```text
model: vit_base_patch16_224
num_classes: 1000
epochs: 90 or pilot 30
batch_size: 256
lr: 1e-3
warmup_epochs: 5 for 90 epochs; maybe 5 or proportionally 3 for 30 epoch pilot
min_lr: 1e-5
weight_decay: 0.07
mixup: 0.2
cutmix: 0.2
mixup_off_epoch: 70 for 90 epochs; around 23 for 30 epochs
smoothing: 0.1
reprob: 0.1
drop_path: 0.1
kfac_damping: 5e-5
kfac_muon_eps: 0.012
kfac_momentum: 0.9
kfac_stats_update_every: 2
kfac_factor_update_every: 2
kfac_ema_decay: 0.95
include first/last layers
static damping
inverse factors bf16
```

---

# Missing Experiments / Open Questions

## Highest Priority

1. **Replicate main results with more seeds.**
   - At least seeds 11, 12, 13 if compute allows.
   - Need mean/std for CIFAR-100 and ImageNet-100.

2. **Recover CIFAR include-first/last args.yaml.**
   - Exact run: `vits16_c100_kfac_lmoff_damp5e5_inclfl_e75_s12`.
   - Summary exists locally, args do not.

3. **Runtime/memory table.**
   - Especially for final ImageNet-100 b256 Muon vs KFAC.

4. **FISMO-style factor comparison or remove/soften claim.**
   - Current intro mentions comparison to FISMO-style variants, but final controlled experiment is not done.

## Medium Priority

5. Full ImageNet-1K pilot.
6. ResNet sanity comparison if paper needs architecture diversity.
7. Test whether adaptive damping helps once final static recipe is established.
8. More detailed diagnostics: factor spectra, damping percentiles, step norms.

---

# What Not To Overclaim

Do not claim:

- Multi-seed robustness yet.
- State-of-the-art ViT training.
- Faster wall-clock training overall unless runtime table supports it.
- Full ImageNet results unless run is completed.
- FISMO comparison unless experiment is done.
- Generality beyond tested ViT image classification unless caveated.

Safe claims:

- KFAC-Muon improves tuned Muon baselines in preliminary single-seed ViT experiments on CIFAR-100 and ImageNet-100.
- The method is naturally motivated as Muon in KFAC-whitened coordinates.
- Including patch embedding and classifier head is important for ViT performance.
- KFAC-Muon reaches higher top-1 and lower validation loss under matched ImageNet-100 b256 recipe.

---

# Suggested Abstract Draft

Here is a possible abstract to adapt:

> Muon has recently emerged as a strong optimizer for matrix-valued neural-network parameters by replacing raw gradient updates with approximately orthogonal polar updates. Separately, natural-gradient and KFAC methods use curvature information to measure steps in the geometry of the model's predictive distribution. We propose KFAC-Muon, an optimizer that applies Muon orthogonalization in KFAC-whitened coordinates. For affine layers, KFAC-Muon whitens the momentum using activation and output-gradient Kronecker factors, applies a Newton-Schulz polar iteration in the whitened space, and maps the update back to parameter coordinates. This can be interpreted as a spectral-norm trust-region update under an approximate Fisher geometry. We also describe a KFAC-reduce approximation for shared-weight layers such as ViT patch embeddings and transformer linear projections. In preliminary single-seed experiments, KFAC-Muon improves over tuned Muon baselines on CIFAR-100 with ViT-S/16 and ImageNet-100 with ViT-B/16, improving top-1 accuracy by 2.17 and 1.76 points respectively while reducing validation loss. These results suggest that Fisher-whitened orthogonal updates can improve the update efficiency of Muon, with layer coverage and damping playing important practical roles.

Need to adjust once multi-seed results are available.

---

# Suggested Introduction Ending / Contributions

Potential contribution wording:

```latex
Our contributions are:
\begin{itemize}
    \item We derive KFAC-Muon as a whiten--polar--unwhiten update: Muon applied under a Kronecker-factored approximation to Fisher geometry.
    \item We describe a KFAC-reduce estimator for shared-weight affine layers, allowing the method to be applied to ViT patch embeddings and transformer-style linear maps.
    \item We implement KFAC-Muon with practical damping, factor-update, and inverse-factor approximations suitable for modern GPU training.
    \item In preliminary ViT image-classification experiments, KFAC-Muon improves over tuned Muon baselines on CIFAR-100 and ImageNet-100, with gains of 2.17 and 1.76 top-1 points respectively.
\end{itemize}
```

If FISMO comparison is added later, include it as a fifth contribution.

---

# Related Work Pointers

Need citations/bibliography cleanup. Current PDF shows unresolved citation keys inline.

Likely citations:

- Muon original / Keller Jordan modded-nanogpt / Muon docs, depending on exact citation source.
- Practical Muon / Muon theory papers referenced in draft as `jordan2024muon`, `liu2025muon`, `shah2025practical`.
- Natural gradient: Amari 1998.
- KFAC: Martens and Grosse 2015; Martens 2020 natural gradient review.
- FISMO: `xu2026fismo` as referenced in draft.
- ViT: Dosovitskiy et al. 2020/2021.
- DeiT: Touvron et al. 2020/2021 for ImageNet-only ViT training reference.
- Possibly Shampoo/SOAP/EKFAC as related Kronecker/preconditioned optimizers.

Do not invent BibTeX. Ask user or search official/arXiv pages if exact bibliography is needed.

---

# Current Local Files Of Interest

Repo:

```text
/Users/luke/kfac muon
```

Current write-up fragments/artifacts:

```text
/Users/luke/kfac muon/notes/experiments_section.tex
/Users/luke/kfac muon/notes/kfac_muon_paper_drafting_handoff.md
/Users/luke/kfac muon/notes/make_experiment_figures.py
/Users/luke/kfac muon/notes/figures/
/Users/luke/kfac muon/notes/tables/main_results.csv
```

Current PDF supplied by user:

```text
/Users/luke/Downloads/kfac_muon (1).pdf
```

Important summaries:

```text
/Users/luke/Downloads/vits16_c100_partial/summaries/vits16_c100_kfac_lmoff_damp5e5_inclfl_e75_s12/summary.csv
/Users/luke/Downloads/vits16_c100_last_sweep/summaries/vits16_c100_muon_lr1p2e3_e75_s12/summary.csv
/Users/luke/Downloads/kfac_b256_e90/summary.csv
/Users/luke/Downloads/muon_b256_e90/summary.csv
```

Important args:

```text
/Users/luke/Downloads/vits16_c100_last_sweep/args/vits16_c100_muon_lr1p2e3_e75_s12/args.yaml
```

Missing args:

```text
/Users/luke/Downloads/vits16_c100_partial/args/vits16_c100_kfac_lmoff_damp5e5_inclfl_e75_s12/args.yaml
```

Need to recover from Vast if possible.

---

# Concrete Next Steps For The User

## Paper Writing

1. Clean structure and remove duplicate scaffold sections.
2. Finish derivation of whiten-polar-unwhiten update.
3. Write KFAC-reduce section.
4. Rewrite contributions to match actual completed experiments.
5. Clean citations/bibliography.
6. Keep experiment section as preliminary until multi-seed results are done.

## Experiments

1. Recover CIFAR KFAC args.
2. Run additional seeds for main CIFAR and ImageNet-100 comparisons.
3. Add matched runtime/memory table.
4. Decide whether to run FISMO-style factor comparison or remove the claim.
5. Consider full ImageNet-1K 30-epoch paired pilot.

## Most Important Recommendation

Do not do more hyperparameter wandering right now. The best next value is:

1. verification/replication,
2. runtime accounting,
3. derivation clarity,
4. optional full ImageNet pilot.

The project has a coherent story already. The next phase is to make it reliable and paper-shaped.
