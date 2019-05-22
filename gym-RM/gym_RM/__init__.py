from gym.envs.registration import register

default_micro_times = 500
default_capacity = 50
default_actions = tuple(k for k in range(50, 231, 20))

# kwargs = {'micro_times': default_micro_times, 'capacity': default_capacity, 'actions': default_actions}

register(
    id='RM-v0',
    entry_point='gym_RM.envs:RMEnv',

)
