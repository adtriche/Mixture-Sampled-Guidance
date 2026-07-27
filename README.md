# Mixture-Sampled-Guidance
This repository contains the complete experimental apparatus for Separating the Credit and Occupancy Effects of Intrinsic Motivation in Policy-Gradient Learning, which introduces Mixture Sampled Guidance (MSG): a decoupled exploration framework whose behavior policy is a mixture of an extrinsic-only actor and an intrinsically motivated advisor, so that intrinsic reward influences the actor only through the training data rather than through its update.

The experiments run in two tiers. In enumerable gridworlds, every quantity the paper's analysis names, occupancies, advantages, the mean mixture disagreement, and the withdrawal sensitivity, is computed exactly by linear solves under linear-softmax policies (exp_*.py, driven by apparatus.py, panel.py, and rollout.py). In MiniGrid (DoorKey and FourRooms), convolutional actor-critics train by PPO from pixel observations with an unmodified RND intrinsic signal (tier2_*.py), comparing MSG against reward blending, an annealed variant, and an extrinsic-only baseline under identical budgets and seeds.

If you use this code, please cite the paper (citation added upon publication).
