from .sac import SAC, SACExpert
from .ppo import PPO, PPOExpert
from .base import Algorithm
from .airl_ppo import AIRLPPO
from .gail import GAIL
from .bc import BC

ALGOS = {"airl_ppo": AIRLPPO, "gail": GAIL, "bc": BC}
