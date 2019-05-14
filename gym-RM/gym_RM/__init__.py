from gym.envs.registration import register

register(
    id='RM-v0',
    entry_point='gym_RM.envs:RMEnv',
)