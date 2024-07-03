import pandas as pd

# Load the reward data from the files
file_path_1 = 'reward_local_1.txt'
file_path_2 = 'reward_local_2.txt'

rewards_1 = pd.read_csv(file_path_1, header=None)
rewards_2 = pd.read_csv(file_path_2, header=None)

# Calculate the moving average with a window size of 10 for both reward sets
window_size = 5
rewards_1_ma = rewards_1.rolling(window=window_size).mean()
rewards_2_ma = rewards_2.rolling(window=window_size).mean()

import matplotlib.pyplot as plt

# Plot the moving average of rewards for both sets
plt.figure(figsize=(14, 7))
plt.plot(rewards_1_ma, label='Agent 1 - Moving Average')
plt.plot(rewards_2_ma, label='Agent 2 - Moving Average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Moving Average of Rewards')
plt.legend()
plt.show()

# Load the new reward data from the file with two columns for agents
file_path = 'rewards.txt'

# Read the data into a pandas dataframe
rewards = pd.read_csv(file_path, header=None, names=['Agent 1', 'Agent 2'])

# Calculate the moving average with a smoother window size, let's use 50 for better smoothing
smooth_window_size = 5
rewards_smooth_ma = rewards.rolling(window=smooth_window_size).mean()

# Plot the smoothed moving average of rewards for both agents
plt.figure(figsize=(14, 7))
plt.plot(rewards_smooth_ma['Agent 1'], label='Agent 1 - Smoothed Moving Average')
plt.plot(rewards_smooth_ma['Agent 2'], label='Agent 2 - Smoothed Moving Average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Smoothed Moving Average of Rewards for Both Agents')
plt.legend()
plt.show()
