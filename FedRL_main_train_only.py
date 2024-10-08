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
    parser.add_argument('--federated_rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--local_episodes', type=int, default=200, help='Number of local episodes')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--max_reward', type=float, default=11, help='Maximum reward')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save the trained models')
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
    
    # Use any environment to get the state and action dimensions
    env = list(train_envs.values())[0]
    state_dim = env.observation_space.shape[0]
    discrete_action_dim = env.discrete_action_space.n
    continuous_action_dim = env.continuous_action_space.shape[0]
    max_action = float(env.continuous_action_space.high[0])
    
    return train_envs, state_dim, discrete_action_dim, continuous_action_dim, max_action

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

        if episode_reward >= max_reward:
            consecutive_max_episodes += 1
        else:
            consecutive_max_episodes = 0
        
        if consecutive_max_episodes >= patience:
            # Convergence achieved
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

def save_agent(agent, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T.save(agent.get_weights(), save_path)
    print(f"Model saved at {save_path}")

def train_federated(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(args.model_dir, f'global_agent_{current_time}.pt')

    # Initialize environments
    train_envs, state_dim, discrete_action_dim, continuous_action_dim, max_action = initialize_environments(args.users, args.servers)

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

    # Save the global agent after training
    save_agent(global_agent, model_save_path)

    return rewards_train, useful_data_train, global_agent

def plot_training_test_rewards(rewards_train):
    plt.figure(figsize=(12, 6))
    for dag_type, rewards in rewards_train.items():
        plt.plot(rewards, label=f'{dag_type} Train')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Training Rewards for Different DAG Types')
    plt.legend()
    plt.savefig('results/training_rewards.png')
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

    # Train Federated Model
    rewards_train, useful_data_train, global_agent = train_federated(args)

    # Save results and plot as necessary
    plot_training_test_rewards(rewards_train)
    plot_useful_data_training(useful_data_train)

if __name__ == "__main__":
    main()