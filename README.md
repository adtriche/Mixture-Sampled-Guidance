# Mixture-Sampled-Guidance
This repository contains the complete experimental apparatus for Separating the Credit and Occupancy Effects of Intrinsic Motivation in Policy-Gradient Learning, which introduces Mixture Sampled Guidance (MSG): a decoupled exploration framework whose behavior policy is a mixture of an extrinsic-only actor and an intrinsically motivated advisor, so that intrinsic reward influences the actor only through the training data rather than through its update.

The experiments run in two tiers. In enumerable gridworlds, every quantity the paper's analysis names, occupancies, advantages, the mean mixture disagreement, and the withdrawal sensitivity, is computed exactly by linear solves under linear-softmax policies (exp_*.py, driven by apparatus.py, panel.py, and rollout.py). In MiniGrid (DoorKey and FourRooms), convolutional actor-critics train by PPO from pixel observations with an unmodified RND intrinsic signal (tier2_*.py), comparing MSG against reward blending, an annealed variant, and an extrinsic-only baseline under identical budgets and seeds.

All run logs and metadata are archived under runs/. The verification scripts verify_section6.py and verify_tier2.py regenerate every quantitative claim in the paper directly from these records; their frozen outputs are included for comparison. plot_section6.py and plot_tier2.py produce all figures from the same loaders.

To reproduce the reported values without retraining: install from requirements.txt, then run the two verifiers. Training reruns end to end with the seeds recorded in each experiment's meta file; dependencies are pinned (minigrid 3.1.0, gymnasium 1.2.3, torch 2.x).

If you use this code, please cite the paper (citation added upon publication).
