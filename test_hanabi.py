import gym
from hanabi.hanabi_agents.dqn_agent import dqnAgent

def env_builder():
    # Parameters of the environment
    data_collection_points = 100
    micro_times = 5
    capacity = 50
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    return gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points,
                    capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


env = env_builder()

# set the number of games to play out
episode_count = 1000

# set the seed
env.seed(1)

# initialize variables
agent = dqnAgent(0, env, 'memory' + str(0) + '.txt')
scores = []
counts = []
fireworks = []
display_fireworks = True
reward = 0
done = False
batch_size = 64
epochs = 4
iters = 25

for episode in range(episode_count):

    observation= env.reset()
    done = False

    while True:

        last_observation = observation

        action = agent.act(observation, reward, done)

        observation, reward, done, info = env.step(action)

        agent.remember(last_observation, action, observation, reward, done)

        if done:
            agent.reset_params()
            if episode % iters == iters - 1:
                agent.train_agent(batch_size, epochs)
                agent.reset_memory()
            break