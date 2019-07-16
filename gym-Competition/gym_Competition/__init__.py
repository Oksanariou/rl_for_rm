from gym.envs.registration import register

register(
    id='Competition-v0',
    entry_point='gym_Competition.envs:CompetitionEnv',
)
