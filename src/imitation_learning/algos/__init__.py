from .sac import SAC, SACExpert, SBSAC
from .ppo import PPO, PPOExpert
from .base import Algorithm
from .airl_ppo import AIRLPPO
from .gail import GAIL
from .bc import BC, BCExpert
from .dagger import DAgger

ALGOS = {"airl_ppo": AIRLPPO, "gail": GAIL, "bc": BC}
