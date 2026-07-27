# Mixture Sampled Guidance

This repository contains the complete experimental apparatus for *Separating the
Credit and Occupancy Effects of Intrinsic Motivation in Policy-Gradient Learning*,
which introduces Mixture Sampled Guidance (MSG): a decoupled exploration framework
whose behavior policy is a mixture of an extrinsic-only actor and an intrinsically
motivated advisor, so that intrinsic reward influences the actor only through the
training data rather than through its update.

## Contents

The experiments run in two tiers. In enumerable gridworlds, every quantity the
paper's analysis names — occupancies, advantages, the mean mixture disagreement,
and the withdrawal sensitivity — is computed exactly by linear solves under
linear-softmax policies. In MiniGrid (DoorKey and FourRooms), convolutional
actor-critics train by PPO from pixel observations with an unmodified RND
intrinsic signal, comparing MSG against reward blending, an annealed variant,
and an extrinsic-only baseline under identical budgets and seeds.

## Getting the data

Download `runs.zip` from the
[latest release](https://github.com/adtriche/Mixture-Sampled-Guidance/releases)
and unzip it at the repository root, producing `runs/`. The archive contains the
training logs, run metadata, and derived measurements for every experiment
reported in the paper.

## Reproducing the reported values

pip install -r requirements.txt
python verify_tier1.py
python verify_tier2.py

The two verification scripts regenerate every quantitative claim in the paper
directly from the archived records and print a claim-by-claim table; the frozen
outputs from our own runs are included for comparison. `plot_tier1.py` and
`plot_tier2.py` produce all figures from the same loaders.

Training reruns end to end with the seeds recorded in each experiment's meta
file. Dependencies are pinned in `requirements.txt` (minigrid 3.1.0,
gymnasium 1.2.3, torch 2.x).

## Script map

| Script(s) | Role | Paper location |
|---|---|---|
| `apparatus.py`, `sampled_apparatus.py`, `panel.py`, `rollout.py`, `recording.py`, `gridworld_env.py` | Tier-1 environments, policies, and instrumentation | — |
| `exp_bridge.py`, `multiseed_bridge.py` | credit-isolation study | Fig. 1a, 1c; §7.1 |
| `exp_capacity.py` | capacity residuals | Table 1 |
| `exp_control.py` | two-room MSG control and FD validation | Fig. 1b; §7.1 |
| `calibration.py` | displacement-bound check over random configurations | §7.1 |
| `exp_credit.py`, `exp_dial.py`, `exp_variance.py` | shaping-weight and crediting studies | §7.1 |
| `tier2_*.py` | the four MiniGrid learners on both tasks | Fig. 2; §7.2 |
| `plot_tier1.py`, `plot_tier2.py` | figure generation | all figures |
| `verify_tier1.py`, `verify_tier2.py` | claim-by-claim regeneration checks | every reported value |

## License and citation

Released under the MIT License. If you use this code, please cite the paper
(citation to be added upon publication).
