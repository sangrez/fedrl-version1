import argparse
import datetime
import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import dqn_td3_v2
import my_env
import my_env_v1
import user_info
import copy
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning for Task Offloading')
    parser.add_argument('--users', type=int, default=4, help='Number of users')
    parser.add_argument('--servers', type=int, default=3, help='Number of servers')
    parser.add_argument('--federated_rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--local_episodes', type=int, default=500, help='Number of local episodes')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--max_reward', type=float, default=5, help='Maximum reward')
    parser.add_argument('--seed', type=int, default=20, help='Random seed')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save the trained models')
    parser.add_argument('--local_episodes_no_FedRL', type=int, default=10000, help='Number of local episodes without Federated Learning')
    return parser.parse_args()

TRAINING_CONFIG = {
    "NUM_DAGS": 5000,
    "TRAIN_RATIO": 0.8,
    "MAJORITY_RATIO": 0.7,
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 1e-4,
    "MIN_BUFFER_SIZE": 1000,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.01,
    "EPSILON_DECAY": 0.995
}

def setup_environment(seed_value):
    T.manual_seed(seed_value)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed_value)
        T.cuda.manual_seed_all(seed_value)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(seed_value)

def initialize_environments(users, servers):
    # dag_types = ['linear', 'branching', 'mixed', 'grid', 'star', 'tree']
    dag_types = ['linear', 'branching', 'mixed']
    train_envs = {}
    for dag_type in dag_types:
        train_envs[dag_type] = my_env_v1.Offloading(users, servers, dag_type)
    
    # Use any environment to get the state and action dimensions
    env = list(train_envs.values())[0]
    state_dim = env.observation_space.shape[0]
    discrete_action_dim = env.discrete_action_space.n
    continuous_action_dim = env.continuous_action_space.shape[0]
    max_action = float(env.continuous_action_space.high[0])
    
    return train_envs, state_dim, discrete_action_dim, continuous_action_dim, max_action

def fill_replay_buffer(agent, env):
    print("Filling replay buffer with random experiences...")
    while len(agent.replay_buffer.storage) < TRAINING_CONFIG["MIN_BUFFER_SIZE"]:
        state = env.reset()
        done = False
        t = 0
        while not done:
            if t == 0:
                discrete_action = 0  # Offload locally at t=0
                continuous_action = np.ones(env.continuous_action_space.shape[0])
            else:
                discrete_action = env.discrete_action_space.sample()
                continuous_action = env.continuous_action_space.sample()

            action_map = user_info.generate_offloading_combinations(env.edge_servers + 1, env.user_devices)
            actual_offloading_decisions = action_map[discrete_action]
            action = list(actual_offloading_decisions) + continuous_action.tolist()

            next_state, reward, done, _, _, _, _, _, _, _ = env.step(t, action)
            agent.add_transition(state, discrete_action, continuous_action, next_state, reward, done)
            state = np.array(next_state, dtype=np.float32)
            t += 1


def train_local(agent, env, episodes, max_reward, patience, global_start_episode):
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
        global_episode = global_start_episode + episode
        state = env.reset()
        done = False
        episode_reward = 0
        t = 0

        # Initialize episode metrics
        episode_metrics = {
            'average_time': 0,
            'average_energy': 0,
            'average_time_local': 0,
            'average_energy_local': 0,
            'average_time_edge': 0,
            'average_energy_edge': 0,
            'average_time_random': 0,
            'average_energy_random': 0
        }
        
        # Calculate epsilon for this episode based on global_episode
        epsilon = max(TRAINING_CONFIG["EPSILON_END"], 
                     TRAINING_CONFIG["EPSILON_START"] * (TRAINING_CONFIG["EPSILON_DECAY"] ** global_episode))
        
        while not done:
            if t == 0:
                discrete_action = 0
                continuous_action = np.ones(env.continuous_action_space.shape[0])
            else:
                if np.random.random() < epsilon:
                    discrete_action = env.discrete_action_space.sample()
                    continuous_action = env.continuous_action_space.sample()
                else:
                    discrete_action, continuous_action = agent.select_action(state)

            action_map = user_info.generate_offloading_combinations(env.edge_servers + 1, env.user_devices)
            actual_offloading_decisions = action_map[discrete_action]
            action = list(actual_offloading_decisions) + continuous_action.tolist()

            next_state, reward, done, _, _, _, _, _, _, _ = env.step(t, action)
            
            # Get metrics from environment
            metrics = env.return_useful_data()
            if metrics[0] is not None:
                episode_metrics['average_time'] += metrics[0]
                episode_metrics['average_energy'] += metrics[1]
                episode_metrics['average_time_local'] += metrics[2]
                episode_metrics['average_energy_local'] += metrics[3]
                episode_metrics['average_time_edge'] += metrics[4]
                episode_metrics['average_energy_edge'] += metrics[5]
                episode_metrics['average_time_random'] += metrics[6]
                episode_metrics['average_energy_random'] += metrics[7]

            agent.add_transition(state, discrete_action, continuous_action, next_state, reward, done)
            state = np.array(next_state, dtype=np.float32)
            t += 1
            episode_reward += reward
            
            if len(agent.replay_buffer.storage) > TRAINING_CONFIG["BATCH_SIZE"]:
                agent.train()

        if t > 0:
            for key in episode_metrics:
                averaged_value = episode_metrics[key] / t
                useful_data_training[key].append(averaged_value)

        episode_rewards.append(episode_reward)

        if episode_reward >= max_reward:
            consecutive_max_episodes += 1
        else:
            consecutive_max_episodes = 0
        
        if consecutive_max_episodes >= patience:
            print(f"Early stopping triggered after {episode + 1} episodes")
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
    model_save_path = os.path.join(args.model_dir, f'global_agent.pt')

    # Initialize environments
    train_envs, state_dim, discrete_action_dim, continuous_action_dim, max_action = initialize_environments(args.users, args.servers)

    # Initialize global agent
    global_agent = dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)

    # Initialize local agents
    local_agents = {dag_type: dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)
                    for dag_type in train_envs.keys()}

    # Fill replay buffers once before training
    print("Filling replay buffers with random experiences for all agents...")
    for dag_type, agent in local_agents.items():
        fill_replay_buffer(agent, train_envs[dag_type])

    # Initialize global episode counter
    global_episode = 0
    total_episodes = args.federated_rounds * args.local_episodes

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
            round_reward, useful_data_training, _ = train_local(
                agent, train_envs[dag_type],
                episodes=args.local_episodes,
                max_reward=args.max_reward,
                patience=args.patience,
                global_start_episode=global_episode
            )
            rewards_train[dag_type].extend(round_reward)
            
            # Append training useful data for each dag_type
            for key in useful_data_training.keys():
                useful_data_train[dag_type][key].extend(useful_data_training[key])

        # Update global_episode counter
        global_episode += args.local_episodes

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

def train_local_independently(args):
    # Initialize environments and independent local agents
    train_envs, state_dim, discrete_action_dim, continuous_action_dim, max_action = initialize_environments(args.users, args.servers)
    
    independent_agents = {dag_type: dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)
                          for dag_type in train_envs.keys()}
    
    rewards_independent = {dag_type: [] for dag_type in train_envs.keys()}
    useful_data_independent = {dag_type: {'average_time': [], 'average_energy': [], 'average_time_local': [],
                                          'average_energy_local': [], 'average_time_edge': [],
                                          'average_energy_edge': [], 'average_time_random': [],
                                          'average_energy_random': []} for dag_type in train_envs.keys()}

    # Fill replay buffers once before training
    print("Filling replay buffers with random experiences for independent agents...")
    for dag_type, agent in independent_agents.items():
        fill_replay_buffer(agent, train_envs[dag_type])

    # Initialize global episode counter
    global_episode = 0
    total_episodes = args.local_episodes_no_FedRL

    for dag_type, agent in independent_agents.items():
        print(f"Training Independent Agent for DAG Type: {dag_type}")
        rewards, useful_data_training, _ = train_local(
            agent, train_envs[dag_type],
            episodes=args.local_episodes_no_FedRL,
            max_reward=args.max_reward,
            patience=args.patience,
            global_start_episode=global_episode
        )
        
        # Save rewards and useful data for comparison
        rewards_independent[dag_type].extend(rewards)
        for key in useful_data_training.keys():
            useful_data_independent[dag_type][key].extend(useful_data_training[key])
        
        # Save the independently trained model for each DAG type
        model_save_path = os.path.join(args.model_dir, f'independent_agent_{dag_type}.pt')
        save_agent(agent, model_save_path)
    
    return rewards_independent, useful_data_independent

def save_rewards_and_useful_data_to_file(rewards_train_federated, useful_data_train_federated, rewards_train_independent, useful_data_train_independent):
    # Save rewards for federated training
    for dag_type, rewards in rewards_train_federated.items():
        with open(f'text_data/rewards_train_federated_{dag_type}_train.txt', 'w') as f:
            for reward in rewards:
                f.write(f"{reward}\n")

    # Save useful data for federated training
    with open('text_data/useful_data_train_federated.txt', 'w') as f:
        f.write("dag_type,average_time,average_energy,average_time_local,average_energy_local,average_time_edge,average_energy_edge,average_time_random,average_energy_random\n")
        for dag_type, data in useful_data_train_federated.items():
            for i in range(len(data['average_time'])):
                f.write(f"{dag_type},{data['average_time'][i]},{data['average_energy'][i]},{data['average_time_local'][i]},{data['average_energy_local'][i]},"
                        f"{data['average_time_edge'][i]},{data['average_energy_edge'][i]},{data['average_time_random'][i]},{data['average_energy_random'][i]}\n")
    
    # Save rewards for independent training
    for dag_type, rewards in rewards_train_independent.items():
        with open(f'text_data/rewards_train_independent_{dag_type}_train.txt', 'w') as f:
            for reward in rewards:
                f.write(f"{reward}\n")

    # Save useful data for independent training
    with open('text_data/useful_data_train_independent.txt', 'w') as f:
        f.write("dag_type,average_time,average_energy,average_time_local,average_energy_local,average_time_edge,average_energy_edge,average_time_random,average_energy_random\n")
        for dag_type, data in useful_data_train_independent.items():
            for i in range(len(data['average_time'])):
                f.write(f"{dag_type},{data['average_time'][i]},{data['average_energy'][i]},{data['average_time_local'][i]},{data['average_energy_local'][i]},"
                        f"{data['average_time_edge'][i]},{data['average_energy_edge'][i]},{data['average_time_random'][i]},{data['average_energy_random'][i]}\n")

def main():
    args = parse_arguments()
    setup_environment(args.seed)

    # Train Federated Model
    print("Starting Federated Learning Training...")
    rewards_train_federated, useful_data_train_federated, global_agent = train_federated(args)

    # Train Independent Local Agents
    print("Starting Independent Training of Local Agents...")
    # rewards_train_independent, useful_data_train_independent = train_local_independently(args)

    # save_rewards_and_useful_data_to_file(rewards_train_federated, useful_data_train_federated, rewards_train_independent, useful_data_train_independent)
    # Save results and plot as necessary
    plot_training_test_rewards(rewards_train_federated)  # Plot for Federated Training
    # plot_training_test_rewards(rewards_train_independent)  # Plot for Independent Training
    plot_useful_data_training(useful_data_train_federated)  # Useful data for Federated
    # plot_useful_data_training(useful_data_train_independent)  # Useful data for Independent

if __name__ == "__main__":
    main()
