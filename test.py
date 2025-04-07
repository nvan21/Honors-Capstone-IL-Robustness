import gymnasium as gym

from imitation_learning.utils.utils import get_config

config = get_config("sac", "Hopper-v5", "modified_reward_hopper")
print(config["xml_file"])
