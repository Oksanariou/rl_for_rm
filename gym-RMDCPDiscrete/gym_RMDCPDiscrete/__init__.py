from gym.envs.registration import register

register(
    id='RMDCPDiscrete-v0',
    entry_point='gym_RMDCPDiscrete.envs:RMDCPDiscreteEnv',
)
