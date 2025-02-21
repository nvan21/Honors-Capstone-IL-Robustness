- It seems like all algorithms are too different to implement a standard training loop in main.py
  - Each algorithm should have its own train method that takes care of its individual training loop
- Instead of using D4RL data, I will generate my own with SAC so that I have access to the expert policy

# Methods each algorithm should have

- init
  - Should this be different from \_\_init\_\_?
  - Responsible for initializing everything in the algorithm
- train
  - This is responsible for each algorithm's training loop
  - Can interact with other methods that are specific to each algorithm
    - These would normally be in utils, but I think they should all be in the same class for readability
  - Should record the time it took to train
- evaluate
  - This takes in a gymnasium environment (which will be setup with a seed already), and evaluates the two different agents trained on the different rewards (learned and true)
  - It should return the cumulative rewards

# Implementation Steps

1. SAC training on all three environments
   - Validate with D4RL expert level performance to ensure that I have an optimal policy
2. Data collection
3. Deep maximum entropy IRL
4. GAIL
5. BIRL
6. Testing

# AIRL Notes

First thing that's needed are expert trajectories

1. Start with a new PPO/SAC policy
2. Take a step in the environment and record the state, policy's action, reward from the environment, mask (False if not done or not truncated), log probabilities of action distribution, next state
   - Continue this loop for the rollout length
3. Once the rollout length is hit, update the discriminator and the policy
   - For the discriminator update, randomly sample _batch size_ trajectories from current policy rollout buffer and from expert demonstrations buffer
   - Calculate the log probabilities of the expert actions
   - Update the discriminator
     - For the discriminator update
