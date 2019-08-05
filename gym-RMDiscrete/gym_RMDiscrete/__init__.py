from gym.envs.registration import register

register(
    id='RMDiscrete-v0',
    entry_point='gym_RMDiscrete.envs:RMDiscreteEnv',
)
