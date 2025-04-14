import os
import argparse
import torch

from imitation_learning.utils.env import make_env
from imitation_learning.utils.utils import visualize_expert
from imitation_learning.network import AIRLDiscrim


def run(args):
    env = make_env(args.env, render_mode="human")

    disc_file = "./logs/Hopper-v5/airl/normal_env-seed0-20250412-1700/model/step10000000/disc.pth"
    disc = AIRLDiscrim(
        state_shape=env.observation_space.shape,
        hidden_units_r=[32],
        hidden_units_v=(32, 32),
        gamma=0.995,
    )
    disc.load_state_dict(torch.load(disc_file))
    # Helper to create a state tensor easily (shape [1, 11])

    def create_state(*args):
        assert len(args) == 11, "Must provide 11 values for Hopper state"
        return torch.tensor([list(args)], dtype=torch.float32)

    # Case 1: Standing Still (relatively neutral state)
    # Expected Reward: Lower than moving forward.
    state_still = create_state(1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Case 2: Moving Forward Moderately Well (good state)
    # Expected Reward: Significantly positive (or higher than Case 1).
    state_moving_fwd = create_state(
        1.25, 0.0, -0.2, 0.4, 0.1, 2.0, 0.0, 0.1, -0.5, 1.0, 0.2
    )

    # Case 3: Moving Forward Very Well (very good state)
    # Expected Reward: Higher than Case 2.
    state_moving_fwd_fast = create_state(
        1.25, 0.0, -0.2, 0.4, 0.1, 4.0, 0.0, 0.1, -0.5, 1.0, 0.2  # Higher x-vel
    )

    # Case 4: Fallen Over (bad state)
    # Expected Reward: Low, likely negative or significantly lower than upright states.
    state_fallen = create_state(
        0.5, 1.0, 0.8, -1.2, 0.5, 0.1, -0.5, 1.5, 0.0, 0.0, 0.0  # Low height, tilted
    )

    # Case 5: Moving Backward (bad state)
    # Expected Reward: Low, likely negative or significantly lower than moving forward.
    state_moving_bwd = create_state(
        1.25, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0  # Negative x-vel
    )

    # Case 6: Low Height but Moving Forward (ambiguous/likely not great)
    # Expected Reward: Lower than Case 2 (moving fwd well).
    state_low_moving_fwd = create_state(
        0.8, 0.3, -0.1, 0.2, 0.0, 1.0, -0.1, 0.5, 0.0, 0.0, 0.0  # Low height
    )

    batched_state = torch.concat(
        (
            state_still,
            state_moving_fwd,
            state_moving_fwd_fast,
            state_fallen,
            state_moving_bwd,
            state_low_moving_fwd,
        )
    )
    print(disc.g(batched_state))

    print("--- Reward Function Sanity Check ---")
    with torch.no_grad():
        reward_still = disc.g(state_still)
        reward_moving_fwd = disc.g(state_moving_fwd)
        reward_moving_fwd_fast = disc.g(state_moving_fwd_fast)
        reward_fallen = disc.g(state_fallen)
        reward_moving_bwd = disc.g(state_moving_bwd)
        reward_low_moving_fwd = disc.g(state_low_moving_fwd)

        # Print results - .item() gets the scalar value from a single-element tensor
        print(f"State Still:           Reward = {reward_still.item():.4f}")
        print(f"State Moving Fwd:      Reward = {reward_moving_fwd.item():.4f}")
        print(f"State Moving Fwd Fast: Reward = {reward_moving_fwd_fast.item():.4f}")
        print(f"State Fallen:          Reward = {reward_fallen.item():.4f}")
        print(f"State Moving Bwd:      Reward = {reward_moving_bwd.item():.4f}")
        print(f"State Low Moving Fwd:  Reward = {reward_low_moving_fwd.item():.4f}")

        print("--- Interpretation ---")
        print("Check if:")
        print(
            "  - Moving forward states get higher rewards than still/fallen/backward states."
        )
        print("  - Faster forward movement gets higher reward.")
        print(
            "  - Fallen/backward states get significantly lower (potentially negative) rewards."
        )
        print("  - The relative order makes sense for the Hopper task objective.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="InvertedPendulum-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
