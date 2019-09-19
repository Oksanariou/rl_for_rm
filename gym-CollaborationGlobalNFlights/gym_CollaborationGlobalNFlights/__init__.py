from gym.envs.registration import register

register(
    id='CollaborationGlobalNFlights-v0',
    entry_point='gym_CollaborationGlobalNFlights.envs:CollaborationGlobalNFlightsEnv',
)
