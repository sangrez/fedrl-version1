import gym
import dqn_td3
from gym import spaces
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import my_env
import user_info

if __name__ == "__main__":
    users = 2
    servers = 3
    env = my_env.Offloading(users, servers)
    state_dim = env.observation_space.shape[0]
    discrete_action_dim = env.discrete_action_space.n
    continuous_action_dim = env.continuous_action_space.shape[0]
    max_action = float(env.continuous_action_space.high[0])

    agent = dqn_td3.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)

    for episode in range(500):
        state = env.reset()
        done = False
        episode_reward = 0
        t = 0
        while not done:
            if t == 0:
                discrete_action = 0
                continuous_action = np.ones(continuous_action_dim)
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
            if agent.replay_buffer.size > 256:
                agent.train()
        
        print(f"Episode {episode} finished with reward: {episode_reward}")
