import argparse
import datetime
import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import dqn_td3_v2
import my_env
import user_info
import copy
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Testing for Task Offloading')
    parser.add_argument('--users', type=int, default=4, help='Number of users')
    parser.add_argument('--servers', type=int, default=3, help='Number of servers')
    parser.add_argument('--test_episodes', type=int, default=100, help='Number of test episodes')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--model_path', type=str, default='models/global_agent_20241004-132843.pt', help='Path to load the trained global model')
    return parser.parse_args()

def setup_environment(seed_value):
    T.manual_seed(seed_value)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed_value)
        T.cuda.manual_seed_all(seed_value)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def initialize_environments(users, servers):
    test_env = my_env.Offloading(users, servers, 'combined', split='test')

    # Get state and action dimensions from the test environment
    state_dim = test_env.observation_space.shape[0]
    discrete_action_dim = test_env.discrete_action_space.n
    continuous_action_dim = test_env.continuous_action_space.shape[0]
    max_action = float(test_env.continuous_action_space.high[0])

    return test_env, state_dim, discrete_action_dim, continuous_action_dim, max_action

def test_agent(agent, env, num_episodes):
    agent.eval()
    total_rewards = []
    useful_data = {
        'average_time': [],
        'average_energy': [],
        'average_time_local': [],
        'average_energy_local': [],
        'average_time_edge': [],
        'average_energy_edge': [],
        'average_time_random': [],
        'average_energy_random': []
    }

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        t = 0
        
        # Initialize accumulators for each useful data type
        total_average_time = 0
        total_average_energy = 0
        total_average_time_local = 0
        total_average_energy_local = 0
        total_average_time_edge = 0
        total_average_energy_edge = 0
        total_average_time_random = 0
        total_average_energy_random = 0

        while not done:
            if t == 0:
                discrete_action = 0
                continuous_action = np.ones(env.continuous_action_space.shape[0])
            else:
                with T.no_grad():
                    discrete_action, continuous_action = agent.select_action(state)

            action_map = user_info.generate_offloading_combinations(env.edge_servers + 1, env.user_devices)
            actual_offloading_decisions = action_map[discrete_action]
            action = list(actual_offloading_decisions) + continuous_action.tolist()

            next_state, reward, done, _, _, _, _, _, _, _ = env.step(t, action)
            average_time, average_energy, average_time_local, average_energy_local, \
            average_time_edge, average_energy_edge, average_time_random, average_energy_random, \
            _, _, _, _, _, _, _, _ = env.return_useful_data()

            # Accumulate the values over the episode
            total_average_time += average_time
            total_average_energy += average_energy
            total_average_time_local += average_time_local
            total_average_energy_local += average_energy_local
            total_average_time_edge += average_time_edge
            total_average_energy_edge += average_energy_edge
            total_average_time_random += average_time_random
            total_average_energy_random += average_energy_random

            episode_reward += reward
            state = next_state
            t += 1
        
        # Compute the average values for this episode
        if t > 0:  # Avoid division by zero
            useful_data['average_time'].append(total_average_time / t)
            useful_data['average_energy'].append(total_average_energy / t)
            useful_data['average_time_local'].append(total_average_time_local / t)
            useful_data['average_energy_local'].append(total_average_energy_local / t)
            useful_data['average_time_edge'].append(total_average_time_edge / t)
            useful_data['average_energy_edge'].append(total_average_energy_edge / t)
            useful_data['average_time_random'].append(total_average_time_random / t)
            useful_data['average_energy_random'].append(total_average_energy_random / t)

        total_rewards.append(episode_reward)

    return total_rewards, useful_data

def test_model(args):
    setup_environment(args.seed)

    # Initialize test environment
    test_env, state_dim, discrete_action_dim, continuous_action_dim, max_action = initialize_environments(args.users, args.servers)

    # Initialize global agent
    global_agent = dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)

    # Load the pre-trained model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file '{args.model_path}' not found. Please provide the correct path.")
    loaded_weights = T.load(args.model_path)
    if not isinstance(loaded_weights, dict):
        raise ValueError("Loaded model weights are not in the expected format. Please ensure the correct model file is provided.")
    global_agent.set_weights(loaded_weights)
    print(f"Loaded global model from {args.model_path}")

    # Test the global agent and collect useful data
    global_rewards_test, useful_data = test_agent(global_agent, test_env, num_episodes=args.test_episodes)

    # Save test results
    save_rewards_and_useful_data_to_file(global_rewards_test, useful_data)

    print("Testing complete, results saved to 'text_data/global_test_rewards.txt'")

    return global_rewards_test, useful_data

def save_rewards_and_useful_data_to_file(global_rewards_test, useful_data):
    with open('text_data/global_rewards_test.txt', 'w') as f:
        for reward in global_rewards_test:
            f.write(f"{reward}\n")

    # Save useful data for testing
    with open('text_data/useful_data_test.txt', 'w') as f:
        f.write("average_time,average_energy,average_time_local,average_energy_local,average_time_edge,average_energy_edge,average_time_random,average_energy_random\n")
        for i in range(len(useful_data['average_time'])):
            f.write(f"{useful_data['average_time'][i]},{useful_data['average_energy'][i]},{useful_data['average_time_local'][i]},{useful_data['average_energy_local'][i]}," +
                    f"{useful_data['average_time_edge'][i]},{useful_data['average_energy_edge'][i]},{useful_data['average_time_random'][i]},{useful_data['average_energy_random'][i]}\n")

def plot_useful_data(useful_data):
    episodes = range(len(useful_data['average_time']))

    plt.figure(figsize=(12, 6))

    # Plot average time
    plt.subplot(2, 1, 1)
    plt.plot(episodes, useful_data['average_time'], label='Average Time')
    plt.plot(episodes, useful_data['average_time_local'], label='Local Average Time')
    plt.plot(episodes, useful_data['average_time_edge'], label='Edge Average Time')
    plt.plot(episodes, useful_data['average_time_random'], label='Random Average Time')
    plt.xlabel('Episodes')
    plt.ylabel('Time')
    plt.legend()
    plt.title('Average Time over Episodes')

    # Plot average energy
    plt.subplot(2, 1, 2)
    plt.plot(episodes, useful_data['average_energy'], label='Average Energy')
    plt.plot(episodes, useful_data['average_energy_local'], label='Local Average Energy')
    plt.plot(episodes, useful_data['average_energy_edge'], label='Edge Average Energy')
    plt.plot(episodes, useful_data['average_energy_random'], label='Random Average Energy')
    plt.xlabel('Episodes')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Average Energy over Episodes')

    plt.tight_layout()
    plt.savefig('results/useful_data_plot.png')
    plt.close()

def main():
    args = parse_arguments()
    global_rewards_test, useful_data = test_model(args)  # Test the model
    plot_useful_data(useful_data)  # Plot useful data after testing

if __name__ == "__main__":
    main()