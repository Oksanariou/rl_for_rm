from gym_RMDCP.envs.RMDCP_env import RMDCPEnv
from gym.envs.registration import register

register(
    id='RMDCP-v0',
    entry_point='gym_RMDCP.envs:RMDCPEnv',
)