from .sac import SAC, SACExpert
from .base import Algorithm
from .airl_ppo import AIRLPPO
from .airl_iql import AIRLIQL

ALGOS = {"airl_ppo": AIRLPPO, "airl_iql": AIRLIQL}
