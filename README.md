# From Demonstrations to Adaptations: Assessing Imitation Learning Robustness and Learned Reward Transferability

This repository is associated with the research paper: "From Demonstrations to Adaptations: Assessing Imitation Learning Robustness and Learned Reward Transferability" by Nathan Van Utrecht and Cody Fleming (May 2025, Honors Capstone Project).

**[Read the Full Paper (PDF)](assets/paper.pdf)**

## Abstract

Imitation Learning (IL) and Inverse Reinforcement Learning (IRL) offer promising alternatives to manual reward engineering in complex sequential decision-making tasks. However, the robustness of learned policies to environmental perturbations and the transferability of learned rewards remain critical challenges. This paper presents a comprehensive empirical study comparing Behavioral Cloning (BC), Generative Adversarial Imitation Learning (GAIL), and Adversarial Inverse Reinforcement Learning (AIRL) across four MuJoCo continuous control environments subjected to various dynamics and task modifications. We evaluate baseline imitation performance, zero-shot policy robustness, and the effectiveness of transferring AIRL-learned rewards to train new Soft Actor-Critic (SAC) agents in modified settings. Our findings indicate that while AIRL generally exhibits superior zero-shot robustness compared to BC and GAIL, all direct IL policies degrade significantly under substantial shifts. Furthermore, the success of AIRL reward transfer is highly context-dependent: it enables near-oracle performance in some modified environments (e.g., Hopper) but fails in others (e.g., Pusher goal shifts or Ant morphological changes), particularly when demonstrations lack diversity covering the test-time alterations. These results highlight current limitations and guide future research towards developing more broadly generalizable and robust imitation learning systems.

## Research Overview

This research investigates the capabilities of prominent imitation learning algorithms when faced with changes in their operating environments. Our study focuses on two key dimensions:

1.  **Zero-Shot Robustness:** How well do policies learned through Behavioral Cloning (BC), Generative Adversarial Imitation Learning (GAIL), and Adversarial Inverse Reinforcement Learning (AIRL) maintain performance when deployed in environments with altered dynamics or task objectives, without any retraining?
2.  **Reward Transferability:** Can the reward function learned by AIRL in a base environment be effectively used to train a new reinforcement learning agent (Soft Actor-Critic) from scratch in these modified environments?

We conduct experiments across four continuous control benchmarks (InvertedPendulum, Hopper, Pusher, and Ant) using a range of modifications, including changes to physical parameters (mass, friction, gravity) and task goals.

## Key Contributions and Findings

The primary contributions of this work include:

*   A systematic baseline evaluation of BC, GAIL, and AIRL policies against expert performance in standard MuJoCo environments.
*   An empirical assessment of the zero-shot robustness of these IL policies when transferred to perturbed environments, identifying relative strengths and weaknesses.
*   An investigation into the efficacy of transferring AIRL-learned reward functions to train new agents in modified settings, highlighting scenarios of both successful adaptation and failure.
*   A comparative analysis contrasting the performance of zero-shot IL policies against agents retrained using transferred AIRL rewards, revealing conditions under which reward transfer offers significant advantages.

Our findings indicate that while AIRL can offer improved robustness and its learned rewards can facilitate adaptation in some cases, significant challenges remain in achieving consistent generalization, particularly when modifications are severe or differ substantially from the training distribution of expert demonstrations.

## Environments Studied

The study utilizes the following MuJoCo continuous control environments from the Gymnasium suite:
*   InvertedPendulum-v5
*   Hopper-v5
*   Pusher-v5
*   Ant-v5

Each environment was subjected to specific modifications to its dynamics or task objectives. Detailed descriptions of these environments and the implemented perturbations can be found in Appendix B of the full paper.


