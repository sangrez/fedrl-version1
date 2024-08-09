import gym
# import dqn_td3
# import dqn_td31
import dqn_td3_v2
from gym import spaces
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import my_env
import user_info
from matplotlib import pyplot as plt
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Seed setting
seed_value = 0
T.manual_seed(seed_value)
if T.cuda.is_available():
    T.cuda.manual_seed(seed_value)
    T.cuda.manual_seed_all(seed_value)

T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False

if __name__ == "__main__":
    users = 3
    servers = 3
    env = my_env.Offloading(users, servers)
    state_dim = env.observation_space.shape[0]
    discrete_action_dim = env.discrete_action_space.n
    continuous_action_dim = env.continuous_action_space.shape[0]
    max_action = float(env.continuous_action_space.high[0])

    agent = dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)
    rewards = []
    for episode in range(5000):
        state = env.reset()
        done = False
        episode_reward = 0
        t = 0
        while not done:
            if t == 0:
                discrete_action = 0
                continuous_action = np.ones(env.continuous_action_space.shape[0])
            else:
                discrete_action, continuous_action = agent.select_action(state)
            action_map = user_info.generate_offloading_combinations(servers + 1, users)
            actual_offloading_decisions = action_map[discrete_action]
            action = list(actual_offloading_decisions) + continuous_action.tolist()

            next_state, reward, done, _ = env.step(t, action, user_info.user_info(users))
            agent.add_transition(state, discrete_action, continuous_action, next_state, reward, done)
            state = np.array(next_state, dtype=np.float32)
            t += 1
            episode_reward += reward
            if len(agent.replay_buffer.storage) > 256:
                agent.train()
        rewards.append(episode_reward)
        print(f"Episode {episode} finished with reward: {episode_reward}")


plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.xlabel('Federated Learning Rounds')
plt.ylabel('Rewards')
plt.title('Rewards for both agents during Federated Learning')
plt.legend(['local_agent'])
plt.grid(True)
plt.savefig(f"results/{current_time}.png")
plt.show()
