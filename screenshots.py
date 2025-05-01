import gymnasium as gym
import matplotlib.pyplot as plt
import os
import numpy as np  # For creating zero-vector actions
import traceback

# --- Configuration ---
# Specific environments requested
env_ids = [
    "Ant-v5",
    "Pusher-v5",
    "Hopper-v5",
    "InvertedPendulum-v5",
]

# Directory to save the screenshots
output_dir = "assets/screenshots"
os.makedirs(output_dir, exist_ok=True)

# Number of steps to simulate after reset for settling
# You might need to adjust this slightly per environment if needed,
# but 10-20 is often a good range for MuJoCo tasks.
SETTLE_STEPS = 15

# --- Prerequisites ---
print("---")
print("IMPORTANT: Ensure you have MuJoCo installed and setup.")
print("You likely need to run: pip install gymnasium[mujoco]")
print("---\n")

# --- Main Loop ---
print(f"Attempting to generate settled screenshots for {len(env_ids)} environments...")

for env_id in env_ids:
    print(f"\nProcessing: {env_id}")
    env = None  # Initialize env to None for the finally block
    try:
        # 1. Create the environment with 'rgb_array' render mode
        # MuJoCo envs generally support rgb_array well.
        env = gym.make(env_id, render_mode="rgb_array")

        # 2. Reset the environment to get the initial state
        # Using a fixed seed ensures the *exact* same starting frame if the env is stochastic
        observation, info = env.reset(seed=1000)

        # --- Settling Phase ---
        # Create a neutral action (vector of zeros for MuJoCo continuous control)
        neutral_action = np.zeros(env.action_space.shape)

        print(f"  Simulating {SETTLE_STEPS} steps with neutral action for settling...")
        terminated = False
        truncated = False
        for i in range(SETTLE_STEPS):
            # Take a step with no action applied
            obs, reward, terminated, truncated, info = env.step(neutral_action)
            # Optional: Render each settling step if you want to debug/visualize it
            # temp_frame = env.render()
            # plt.imsave(os.path.join(output_dir, f"{env_id}_settle_step_{i}.png"), temp_frame)

            if terminated or truncated:
                print(
                    f"  Warning: Environment terminated or truncated after {i+1} settling steps."
                )
                print(
                    "  The resulting screenshot might not represent a stable settled state."
                )
                # Depending on the env, it might be fine, or it might mean it fell over.
                # We'll still capture the frame just before termination/truncation.
                break  # Exit the settling loop

        # 3. Render the frame *after* settling steps
        print("  Rendering final settled frame...")
        frame = env.render()

        if frame is None:
            print(
                f"  Warning: env.render() returned None for {env_id} after settling. Skipping."
            )
            continue  # Skip if rendering didn't produce an array

        # 4. Save the frame as an image
        # Create a filename friendly version of the env_id
        safe_env_id = env_id.replace("/", "_").replace(":", "_")
        output_path = os.path.join(output_dir, f"{safe_env_id}.png")

        plt.imsave(output_path, frame)
        print(f"  Successfully saved screenshot to: {output_path}")

    except ImportError as e:
        print(f"  Skipping {env_id}: Missing dependency - {e}")
        print(
            f"  Make sure you have installed MuJoCo and the gym wrapper: pip install gymnasium[mujoco]"
        )
    except Exception as e:
        print(f"  ERROR processing {env_id}: {e}")
        print("  Traceback:")
        traceback.print_exc()  # Print full traceback for easier debugging
    finally:
        # 5. Clean up: Close the environment
        if env is not None:
            env.close()
            print("  Environment closed.")

print("\nScreenshot generation complete.")
