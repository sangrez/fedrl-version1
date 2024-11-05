import argparse
import datetime
import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import dqn_td3_v2
import my_env
import my_env_v1
import user_info
import copy
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Testing for Task Offloading')
    parser.add_argument('--users', type=int, default=4, help='Number of users')
    parser.add_argument('--servers', type=int, default=3, help='Number of servers')
    parser.add_argument('--test_episodes', type=int, default=100, help='Number of test episodes')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--model_path', type=str, default='models/global_agent.pt', help='Path to load the trained global model')
    parser.add_argument('--local_model_dir', type=str, default='models', help='Directory to load independently trained local models')
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
    dag_types = ['linear', 'branching', 'mixed', 'grid', 'star', 'tree']
    test_envs = {}
    for dag_type in dag_types:
        test_envs[dag_type] = my_env_v1.Offloading(users, servers, dag_type, split='test')
    
    # Use any environment to get the state and action dimensions
    env = list(test_envs.values())[0]
    state_dim = env.observation_space.shape[0]
    discrete_action_dim = env.discrete_action_space.n
    continuous_action_dim = env.continuous_action_space.shape[0]
    max_action = float(env.continuous_action_space.high[0])
    
    return test_envs, state_dim, discrete_action_dim, continuous_action_dim, max_action


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

    # Initialize test environments for all DAG types
    test_envs, state_dim, discrete_action_dim, continuous_action_dim, max_action = initialize_environments(args.users, args.servers)

    # Initialize global agent
    global_agent = dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)

    # Load the pre-trained global model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file '{args.model_path}' not found. Please provide the correct path.")
    loaded_weights = T.load(args.model_path)
    if not isinstance(loaded_weights, dict):
        raise ValueError("Loaded model weights are not in the expected format. Please ensure the correct model file is provided.")
    global_agent.set_weights(loaded_weights)
    print(f"Loaded global model from {args.model_path}")

    # Test the global agent on each DAG type and collect useful data
    all_results_global = {}
    for dag_type, env in test_envs.items():
        print(f"Testing Federated Global Agent on {dag_type} DAG...")
        global_rewards_test, useful_data = test_agent(global_agent, env, num_episodes=args.test_episodes)
        all_results_global[dag_type] = {
            'rewards': global_rewards_test,
            'useful_data': useful_data
        }

    # Test the independently trained local agents on each DAG type
    all_results_local = {}
    for local_dag_type in test_envs.keys():
        local_model_path = os.path.join(args.local_model_dir, f'independent_agent_{local_dag_type}.pt')
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model file '{local_model_path}' not found. Please provide the correct path.")
        local_agent = dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)
        local_weights = T.load(local_model_path)
        if not isinstance(local_weights, dict):
            raise ValueError("Loaded model weights are not in the expected format. Please ensure the correct model file is provided.")
        local_agent.set_weights(local_weights)
        print(f"Loaded local model for {local_dag_type} DAG from {local_model_path}")

        # Test the local agent on all DAG types
        for test_dag_type, env in test_envs.items():
            print(f"Testing Local Agent ({local_dag_type}) on {test_dag_type} DAG...")
            local_rewards_test, useful_data = test_agent(local_agent, env, num_episodes=args.test_episodes)
            all_results_local[f'{local_dag_type}_on_{test_dag_type}'] = {
                'rewards': local_rewards_test,
                'useful_data': useful_data
            }

    # Save test results for all DAG types
    save_all_results_to_file(all_results_global, prefix='global')
    save_all_results_to_file(all_results_local, prefix='local')

    print("Testing complete, results saved to 'text_data/'")

    return all_results_global, all_results_local

def save_all_results_to_file(all_results, prefix='global'):
    # Create a directory to store the results if it doesn't exist
    os.makedirs('text_data', exist_ok=True)

    for dag_type, results in all_results.items():
        # Save rewards
        rewards_file = f'text_data/{prefix}_{dag_type}_rewards.txt'
        with open(rewards_file, 'w') as f:
            f.write("Episode,Reward\n")
            for episode, reward in enumerate(results['rewards']):
                f.write(f"{episode},{reward}\n")
        
        # Save useful data
        useful_data_file = f'text_data/{prefix}_{dag_type}_useful_data.txt'
        with open(useful_data_file, 'w') as f:
            f.write("Episode,average_time,average_energy,average_time_local,average_energy_local,average_time_edge,average_energy_edge,average_time_random,average_energy_random\n")
            useful_data = results['useful_data']
            for i in range(len(useful_data['average_time'])):
                f.write(f"{i},{useful_data['average_time'][i]},{useful_data['average_energy'][i]},{useful_data['average_time_local'][i]},{useful_data['average_energy_local'][i]},"
                        f"{useful_data['average_time_edge'][i]},{useful_data['average_energy_edge'][i]},{useful_data['average_time_random'][i]},{useful_data['average_energy_random'][i]}\n")
        
        print(f"Saved results for {dag_type} DAG to '{rewards_file}' and '{useful_data_file}'")

def plot_all_useful_data(all_results, prefix='global'):
    # Create a directory to store the plots if it doesn't exist
    os.makedirs('results', exist_ok=True)

    for dag_type, results in all_results.items():
        useful_data = results['useful_data']
        episodes = range(len(useful_data['average_time']))

        # Create a new figure for each DAG type
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot average time
        ax1.plot(episodes, useful_data['average_time'], label='Average Time')
        ax1.plot(episodes, useful_data['average_time_local'], label='Local Average Time')
        ax1.plot(episodes, useful_data['average_time_edge'], label='Edge Average Time')
        ax1.plot(episodes, useful_data['average_time_random'], label='Random Average Time')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Time')
        ax1.legend()
        ax1.set_title(f'{dag_type} DAG - Average Time ({prefix})')
        
        # Plot average energy
        ax2.plot(episodes, useful_data['average_energy'], label='Average Energy')
        ax2.plot(episodes, useful_data['average_energy_local'], label='Local Average Energy')
        ax2.plot(episodes, useful_data['average_energy_edge'], label='Edge Average Energy')
        ax2.plot(episodes, useful_data['average_energy_random'], label='Random Average Energy')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Energy')
        ax2.legend()
        ax2.set_title(f'{dag_type} DAG - Average Energy ({prefix})')

        plt.tight_layout()
        plt.savefig(f'results/{prefix}_{dag_type}_dag_useful_data_plot.png')
        plt.close()

        print(f"Saved plot for {dag_type} DAG to 'results/{prefix}_{dag_type}_dag_useful_data_plot.png'")

def plot_rewards(all_results, prefix='global'):
    # Create a directory to store the reward plots if it doesn't exist
    os.makedirs('results', exist_ok=True)

    for dag_type, results in all_results.items():
        episodes = range(len(results['rewards']))

        # Create a new figure for rewards
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, results['rewards'], label='Reward', color='b')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.title(f'{dag_type} DAG - Rewards ({prefix})')
        plt.tight_layout()
        plt.savefig(f'results/{prefix}_{dag_type}_rewards_plot.png')
        plt.close()

        print(f"Saved reward plot for {dag_type} DAG to 'results/{prefix}_{dag_type}_rewards_plot.png'")

def main():
    args = parse_arguments()
    all_results_global, all_results_local = test_model(args)  # Test the models on all DAG types
    # plot_all_useful_data(all_results_global, prefix='global') 
    # plot_all_useful_data(all_results_local, prefix='local')
    plot_rewards(all_results_global, prefix='global')
    plot_rewards(all_results_local, prefix='local')

if __name__ == "__main__":
    main()