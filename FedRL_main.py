import argparse
import datetime
import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import dqn_td3_v2
import my_env
import user_info
import copy
from tqdm import tqdm
def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning for Task Offloading')
    parser.add_argument('--users', type=int, default=4, help='Number of users')
    parser.add_argument('--servers', type=int, default=3, help='Number of servers')
    parser.add_argument('--federated_rounds', type=int, default=10, help='Number of federated rounds')
    parser.add_argument('--local_episodes', type=int, default=200, help='Number of local episodes')
    parser.add_argument('--test_episodes', type=int, default=100, help='Number of test episodes')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--max_reward', type=float, default=11, help='Maximum reward')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
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
    dag_types = ['linear', 'branching', 'mixed', 'grid', 'star', 'tree', 'cycle-free-mesh']
    train_envs = {}
    for dag_type in dag_types:
        train_envs[dag_type] = my_env.Offloading(users, servers, dag_type)
    
    # Initialize a single test environment with combined DAGs
    test_env = my_env.Offloading(users, servers, 'comibned', split='test')
    
    # Use any environment to get the state and action dimensions
    env = list(train_envs.values())[0]
    state_dim = env.observation_space.shape[0]
    discrete_action_dim = env.discrete_action_space.n
    continuous_action_dim = env.continuous_action_space.shape[0]
    max_action = float(env.continuous_action_space.high[0])
    
    return train_envs, test_env, state_dim, discrete_action_dim, continuous_action_dim, max_action

def train_local(agent, env, episodes, max_reward=11, patience=50):
    episode_rewards = []
    consecutive_max_episodes = 0
    useful_data_training = {
        'average_time': [],
        'average_energy': [],
        'average_time_local': [],
        'average_energy_local': [],
        'average_time_edge': [],
        'average_energy_edge': [],
        'average_time_random': [],
        'average_energy_random': []
    }

    for episode in tqdm(range(episodes), desc="Training Progress"):
        state = env.reset()
        done = False
        episode_reward = 0
        t = 0

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

            agent.add_transition(state, discrete_action, continuous_action, next_state, reward, done)
            state = np.array(next_state, dtype=np.float32)
            t += 1
            episode_reward += reward
            
            if len(agent.replay_buffer.storage) > 256:
                agent.train()
        if t > 0:  # Avoid division by zero
            useful_data_training['average_time'].append(total_average_time / t)
            useful_data_training['average_energy'].append(total_average_energy / t)
            useful_data_training['average_time_local'].append(total_average_time_local / t)
            useful_data_training['average_energy_local'].append(total_average_energy_local / t)
            useful_data_training['average_time_edge'].append(total_average_time_edge / t)
            useful_data_training['average_energy_edge'].append(total_average_energy_edge / t)
            useful_data_training['average_time_random'].append(total_average_time_random / t)
            useful_data_training['average_energy_random'].append(total_average_energy_random / t)        
    
        episode_rewards.append(episode_reward)
        # print(f"Episode {episode} finished with reward: {episode_reward}")

        if episode_reward >= max_reward:
            consecutive_max_episodes += 1
        else:
            consecutive_max_episodes = 0
        
        if consecutive_max_episodes >= patience:
            # print(f"Convergence achieved after {episode + 1} episodes.")
            break

    return episode_rewards, useful_data_training, consecutive_max_episodes

def federated_averaging(global_agent, agents):
    local_weights = [agent.get_weights() for agent in agents]
    averaged_weights = {
        'dqn': copy.deepcopy(local_weights[0]['dqn']),
        'dqn_target': copy.deepcopy(local_weights[0]['dqn_target']),
        'actor': copy.deepcopy(local_weights[0]['actor']),
        'critic': copy.deepcopy(local_weights[0]['critic']),
        'critic_target': copy.deepcopy(local_weights[0]['critic_target'])
    }
    for key in averaged_weights.keys():
        for subkey in averaged_weights[key].keys():
            averaged_weights[key][subkey] = sum(agent[key][subkey] for agent in local_weights) / len(local_weights)

    global_agent.set_weights(averaged_weights)
    for agent in agents:
        agent.set_weights(averaged_weights)

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

    agent.train()
    return total_rewards, useful_data

def train_and_test(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Initialize environments
    train_envs, test_env, state_dim, discrete_action_dim, continuous_action_dim, max_action = initialize_environments(args.users, args.servers)

    # Initialize global agent
    global_agent = dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)

    # Initialize local agents
    local_agents = {dag_type: dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action) 
                    for dag_type in train_envs.keys()}

    # Train agents
    rewards_train = {dag_type: [] for dag_type in train_envs.keys()}
    useful_data_train = {dag_type: {'average_time': [], 'average_energy': [], 'average_time_local': [],
                                    'average_energy_local': [], 'average_time_edge': [],
                                    'average_energy_edge': [], 'average_time_random': [],
                                    'average_energy_random': []} for dag_type in train_envs.keys()}

    for round in tqdm(range(args.federated_rounds), desc="Federated Rounds Progress"):
        print(f"Federated Learning Round {round + 1}")
        
        # Training
        for dag_type, agent in local_agents.items():
            round_reward, useful_data_training, _ = train_local(agent, train_envs[dag_type], episodes=args.local_episodes, max_reward=args.max_reward, patience=args.patience)
            rewards_train[dag_type].extend(round_reward)
            
            # Append training useful data for each dag_type
            for key in useful_data_training.keys():
                useful_data_train[dag_type][key].extend(useful_data_training[key])

        # Perform federated averaging
        federated_averaging(global_agent, list(local_agents.values()))

    return rewards_train, useful_data_train, global_agent, test_env

def plot_training_test_rewards(rewards_train, global_rewards_test):
    plt.figure(figsize=(12, 6))
    for dag_type, rewards in rewards_train.items():
        plt.plot(rewards, label=f'{dag_type} Train')
    plt.plot(global_rewards_test, label='Global Test', linewidth=2, linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Training Rewards for Different DAG Types and Global Test Rewards')
    plt.legend()
    plt.savefig('results/training_test_rewards.png')
    plt.close()

def save_rewards_and_useful_data_to_file(rewards_train, global_rewards_test, useful_data, useful_data_training):
    for dag_type, rewards in rewards_train.items():
        with open(f'text_data/rewards_{dag_type}_train.txt', 'w') as f:
            for reward in rewards:
                f.write(f"{reward}\n")
    
    with open('text_data/global_rewards_test.txt', 'w') as f:
        for reward in global_rewards_test:
            f.write(f"{reward}\n")

    # Save useful data for testing
    with open('text_data/useful_data_test.txt', 'w') as f:
        f.write("average_time,average_energy,average_time_local,average_energy_local,average_time_edge,average_energy_edge,average_time_random,average_energy_random\n")
        for i in range(len(useful_data['average_time'])):
            f.write(f"{useful_data['average_time'][i]},{useful_data['average_energy'][i]},{useful_data['average_time_local'][i]},{useful_data['average_energy_local'][i]},"
                    f"{useful_data['average_time_edge'][i]},{useful_data['average_energy_edge'][i]},{useful_data['average_time_random'][i]},{useful_data['average_energy_random'][i]}\n")

    # Save useful data for training
    with open('text_data/useful_data_training.txt', 'w') as f:
        f.write("dag_type,average_time,average_energy,average_time_local,average_energy_local,average_time_edge,average_energy_edge,average_time_random,average_energy_random\n")
        for dag_type, data in useful_data_training.items():
            for i in range(len(data['average_time'])):  # Access each DAG type's useful data
                f.write(f"{dag_type},{data['average_time'][i]},{data['average_energy'][i]},{data['average_time_local'][i]},{data['average_energy_local'][i]},"
                        f"{data['average_time_edge'][i]},{data['average_energy_edge'][i]},{data['average_time_random'][i]},{data['average_energy_random'][i]}\n")
          
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

def plot_useful_data_training(useful_data_training):

    for dag_type, data in useful_data_training.items():
        episodes = range(len(data['average_time']))

        plt.figure(figsize=(12, 6))

        # Plot average time
        plt.subplot(2, 1, 1)
        plt.plot(episodes, data['average_time'], label='Average Time')
        plt.plot(episodes, data['average_time_local'], label='Local Average Time')
        plt.plot(episodes, data['average_time_edge'], label='Edge Average Time')
        plt.plot(episodes, data['average_time_random'], label='Random Average Time')
        plt.xlabel('Episodes')
        plt.ylabel('Time')
        plt.legend()
        plt.title(f'Average Time over Episodes ({dag_type})')

        # Plot average energy
        plt.subplot(2, 1, 2)
        plt.plot(episodes, data['average_energy'], label='Average Energy')
        plt.plot(episodes, data['average_energy_local'], label='Local Average Energy')
        plt.plot(episodes, data['average_energy_edge'], label='Edge Average Energy')
        plt.plot(episodes, data['average_energy_random'], label='Random Average Energy')
        plt.xlabel('Episodes')
        plt.ylabel('Energy')
        plt.legend()
        plt.title(f'Average Energy over Episodes ({dag_type})')

        plt.tight_layout()
        plt.savefig(f'results/useful_data_training_{dag_type}.png')
        plt.close()

def main():
    args = parse_arguments()
    setup_environment(args.seed)
    global_rewards_test = []
    
    # Get training rewards, training useful data, and global agent
    rewards_train, useful_data_training, global_agent, test_env = train_and_test(args)
    
    # Test the global agent and collect useful data
    global_test_reward, useful_data = test_agent(global_agent, test_env, num_episodes=args.test_episodes)
    
    # Save the rewards and useful data to files (for both training and testing)
    save_rewards_and_useful_data_to_file(rewards_train, global_test_reward, useful_data, useful_data_training)
    
    # Plot training and test rewards
    plot_training_test_rewards(rewards_train, global_rewards_test)
    
    # Plot useful data for both training and testing
    plot_useful_data_training(useful_data_training)
    plot_useful_data(useful_data)


if __name__ == "__main__":
    main()
