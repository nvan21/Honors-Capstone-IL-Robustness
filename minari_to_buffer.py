import torch
import minari
import numpy as np

from imitation_learning.utils.buffer import Buffer


def load_minari_to_buffer(
    dataset_name, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load a Minari dataset into a Buffer, automatically determining the buffer size.

    Args:
        dataset_name: Name of the Minari dataset (e.g., "atari/breakout/expert-v0")
        device: Torch device to store tensors on

    Returns:
        Populated Buffer instance
    """
    # Load the dataset
    print(f"Loading Minari dataset '{dataset_name}'...")
    dataset = minari.load_dataset(dataset_name, download=True)

    # Count total timesteps
    total_timesteps = 0
    for episode in dataset:
        total_timesteps += len(episode)
    print(
        f"Dataset contains {total_timesteps} total timesteps across {len(dataset)} episodes"
    )

    # Get shapes
    observation_shape = dataset[0].observations[0].shape

    # Determine action shape based on action space type
    if hasattr(dataset.action_space, "shape"):
        # Continuous action space
        action_shape = dataset.action_space.shape
    else:
        # Discrete action space - convert to one-hot later
        action_shape = (1,)

    # Create buffer with exact size needed
    buffer = Buffer(total_timesteps, observation_shape, action_shape, device)

    # Iterate through episodes
    transitions_added = 0
    print(f"Loading transitions into buffer...")

    for episode_idx, episode in enumerate(dataset):
        episode_length = len(episode)

        for t in range(episode_length):
            # Current state and action
            state = episode.observations[t]
            action = episode.actions[t]
            reward = episode.rewards[t]

            # Use terminations and truncations for done flag
            # A step is done if it's either terminated or truncated
            done = False
            if t < len(episode.terminations):
                done = done or episode.terminations[t]
            if t < len(episode.truncations):
                done = done or episode.truncations[t]

            # Handle the next state and done flag
            if t == episode_length - 1:
                # Last timestep in episode
                next_state = np.zeros_like(
                    state
                )  # Or use a terminal state representation
            else:
                # Get next state from the next timestep
                next_state = episode.observations[t + 1]

            # Convert discrete actions to the right format if needed
            if not hasattr(dataset.action_space, "shape") and action_shape == (1,):
                # Convert discrete action to scalar
                action = np.array([action], dtype=np.float32)

            # Add to buffer
            buffer.append(state, action, reward, done, next_state)
            print(transitions_added)
            transitions_added += 1

        # Print progress
        if (episode_idx + 1) % 10 == 0 or episode_idx == len(dataset) - 1:
            print(
                f"Processed {episode_idx + 1}/{len(dataset)} episodes, {transitions_added}/{total_timesteps} transitions"
            )

    print(f"Finished loading dataset with {transitions_added} transitions")
    return buffer


buffer = load_minari_to_buffer("D4RL/pointmaze/large-dense-v2")
buffer.save("datasets/PointMaze_LargeDense-v3.pth")
