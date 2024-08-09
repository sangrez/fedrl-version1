import pandas as pd

# Load the reward data from the files
file_path_1 = 'rewards_branching_train.txt'
file_path_2 = 'rewards_linear_train.txt'
file_path_3 = 'rewards_mixed_train.txt'
file_path_4 = 'global_rewards_branching_val.txt'
file_path_5 = 'global_rewards_linear_val.txt'
file_path_6 = 'global_rewards_mixed_val.txt'

rewards_1 = pd.read_csv(file_path_1, header=None)
rewards_2 = pd.read_csv(file_path_2, header=None)
rewards_3 = pd.read_csv(file_path_3, header=None)
rewards_4 = pd.read_csv(file_path_4, header=None)
rewards_5 = pd.read_csv(file_path_5, header=None)
rewards_6 = pd.read_csv(file_path_6, header=None)
# Calculate the moving average with a window size of 10 for both reward sets
window_size = 500
window_size_val = 1
rewards_1_ma = rewards_1.rolling(window=window_size).mean()
rewards_2_ma = rewards_2.rolling(window=window_size).mean()
rewards_3_ma = rewards_3.rolling(window=window_size).mean()
rewards_4_ma = rewards_4.rolling(window=window_size_val).mean()
rewards_5_ma = rewards_5.rolling(window=window_size_val).mean()
rewards_6_ma = rewards_6.rolling(window=window_size_val).mean()

import matplotlib.pyplot as plt

# Plot the moving average of rewards for both sets
plt.figure(figsize=(14, 7))
plt.plot(rewards_1_ma, label='Agent 1 - Moving Average')
plt.plot(rewards_2_ma, label='Agent 2 - Moving Average')
plt.plot(rewards_3_ma, label='Agent 3 - Moving Average')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Moving Average of Rewards')
plt.legend()
plt.grid(True)
plt.savefig('rewards.png')
# plt.show()

plt.figure(figsize=(14, 7))
plt.plot(rewards_4_ma, label='Agent 1 val - Moving Average')
plt.plot(rewards_5_ma, label='Agent 2 val - Moving Average')
plt.plot(rewards_6_ma, label='Agent 3 val - Moving Average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Smoothed Moving Average of Rewards for Both Agents')
plt.legend()
plt.show()
