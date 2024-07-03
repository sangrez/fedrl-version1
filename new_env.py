import gym
from gym import spaces
import numpy as np
from user_info import parameter
import user_info

class TaskOffloadingEnv(gym.Env):
    def __init__(self, users, servers):

        self.user_devices = users
        self.edge_servers = servers
        

        # state space
        low = np.array( [-50] * self.user_devices +  [0] * self.user_devices +  [0]).astype(np.float32)  
        high = np.array([-5] * self.user_devices +  [100] * self.user_devices +  [100]).astype(np.float32)  
        self.observation_space = spaces.Box(low, high)


        # action space
        all_combinations = user_info.generate_offloading_combinations(self.edge_servers + 1, self.user_devices)
        self.discrete_action_space = spaces.Discrete(len(all_combinations))
        low_cpu = np.array([0.1] * self.user_devices).astype(np.float32)
        high_cpu = np.array([self.max_cpu] * self.user_devices).astype(np.float32)
        self.continuous_action_space = spaces.Box(low=low_cpu, high=high_cpu, dtype=np.float32)
        self.action_space = spaces.Tuple((self.discrete_action_space, self.continuous_action_space))
        
    def reset(self):
        # Reset the environment to its initial state
        # Return the initial observation
        pass

    def step(self, action):
        # Perform the given action and update the environment
        # Return the next observation, reward, done flag, and additional info
        pass

    def render(self, mode='human'):
        # Render the environment
        pass