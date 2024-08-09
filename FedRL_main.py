import argparse
import datetime
import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
from torch.utils.tensorboard import SummaryWriter
import dqn_td3_v2
import my_env
import user_info

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning for Task Offloading')
    parser.add_argument('--users', type=int, default=5, help='Number of users')
    parser.add_argument('--servers', type=int, default=3, help='Number of servers')
    parser.add_argument('--federated_rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--local_episodes', type=int, default=400, help='Number of local episodes')
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

def initialize_environment(users, servers, dag_type, split='train'):
    env = my_env.Offloading(users, servers, dag_type)
    env.reset_eval()  # Reset the environment
    state_dim = env.observation_space.shape[0]
    discrete_action_dim = env.discrete_action_space.n
    continuous_action_dim = env.continuous_action_space.shape[0]
    max_action = float(env.continuous_action_space.high[0])
    return env, state_dim, discrete_action_dim, continuous_action_dim, max_action

def train_local(agent, env, episodes, max_reward=11, patience=50):
    episode_rewards = []
    consecutive_max_episodes = 0

    for episode in range(episodes):
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

            action_map = user_info.generate_offloading_combinations(env.edge_servers + 1, env.user_devices)
            actual_offloading_decisions = action_map[discrete_action]
            action = list(actual_offloading_decisions) + continuous_action.tolist()

            next_state, reward, done, _ = env.step(t, action)
            agent.add_transition(state, discrete_action, continuous_action, next_state, reward, done)
            state = np.array(next_state, dtype=np.float32)
            t += 1
            episode_reward += reward
            
            if len(agent.replay_buffer.storage) > 256:
                agent.train()
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode} finished with reward: {episode_reward}")

        if episode_reward >= max_reward:
            consecutive_max_episodes += 1
        else:
            consecutive_max_episodes = 0
        
        if consecutive_max_episodes >= patience:
            print(f"Convergence achieved after {episode + 1} episodes.")
            break

    return episode_rewards, consecutive_max_episodes

def federated_averaging(global_agent, agents):
    local_weights = [agent.get_weights() for agent in agents]
    averaged_weights = {
        'dqn': copy.deepcopy(local_weights[0]['dqn']),
        'dqn_target': copy.deepcopy(local_weights[0]['dqn_target']),
        'actor': copy.deepcopy(local_weights[0]['actor']),
        'critic': copy.deepcopy(local_weights[0]['critic']),
        'critic_target': copy.deepcopy(local_weights[0]['critic_target'])
    }
    for key in averaged_weights['dqn'].keys():
        for i in range(1, len(local_weights)):
            averaged_weights['dqn'][key] += local_weights[i]['dqn'][key]
        averaged_weights['dqn'][key] /= len(local_weights)
    for key in averaged_weights['dqn_target'].keys():
        for i in range(1, len(local_weights)):
            averaged_weights['dqn_target'][key] += local_weights[i]['dqn_target'][key]
        averaged_weights['dqn_target'][key] /= len(local_weights)
    for key in averaged_weights['actor'].keys():
        for i in range(1, len(local_weights)):
            averaged_weights['actor'][key] += local_weights[i]['actor'][key]
        averaged_weights['actor'][key] /= len(local_weights)
    for key in averaged_weights['critic'].keys():
        for i in range(1, len(local_weights)):
            averaged_weights['critic'][key] += local_weights[i]['critic'][key]
        averaged_weights['critic'][key] /= len(local_weights)
    for key in averaged_weights['critic_target'].keys():
        for i in range(1, len(local_weights)):
            averaged_weights['critic_target'][key] += local_weights[i]['critic_target'][key]
        averaged_weights['critic_target'][key] /= len(local_weights)

    global_agent.set_weights(averaged_weights)

def test_agent(agent, env, num_episodes=100):
    agent.eval()  # Set the agent to evaluation mode
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        t = 0
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
            
            next_state, reward, done, _ = env.step(t, action)
            episode_reward += reward
            state = next_state
            t += 1
        total_rewards.append(episode_reward)
        print(f"Episode {episode} finished with test reward: {episode_reward}")
    agent.train()  # Set the agent back to training mode
    return np.mean(total_rewards)

def train_and_test(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/federated_learning_experiment_{current_time}')

    # Initialize environments
    env_linear_train, state_dim_linear, discrete_action_dim_linear, continuous_action_dim_linear, max_action_linear = initialize_environment(args.users, args.servers, 'linear', 'train')
    env_branching_train, _, _, _, _ = initialize_environment(args.users, args.servers, 'branching', 'train')
    env_mixed_train, _, _, _, _ = initialize_environment(args.users, args.servers, 'mixed', 'train')

    env_linear_val, _, _, _, _ = initialize_environment(args.users, args.servers, 'linear', 'val')
    env_branching_val, _, _, _, _ = initialize_environment(args.users, args.servers, 'branching', 'val')
    env_mixed_val, _, _, _, _ = initialize_environment(args.users, args.servers, 'mixed', 'val')

    # Initialize global agent
    global_agent = dqn_td3_v2.JointAgent(state_dim_linear, discrete_action_dim_linear, continuous_action_dim_linear, max_action_linear)

    # Initialize local agents
    agent_linear = dqn_td3_v2.JointAgent(state_dim_linear, discrete_action_dim_linear, continuous_action_dim_linear, max_action_linear)
    agent_branching = dqn_td3_v2.JointAgent(state_dim_linear, discrete_action_dim_linear, continuous_action_dim_linear, max_action_linear)
    agent_mixed = dqn_td3_v2.JointAgent(state_dim_linear, discrete_action_dim_linear, continuous_action_dim_linear, max_action_linear)

    local_agents = [agent_linear, agent_branching, agent_mixed]
    train_envs = [env_linear_train, env_branching_train, env_mixed_train]
    val_envs = [env_linear_val, env_branching_val, env_mixed_val]

    # Train agents
    rewards_train = {dag_type: [] for dag_type in ['linear', 'branching', 'mixed']}
    rewards_val = {dag_type: [] for dag_type in ['linear', 'branching', 'mixed']}
    global_rewards_val = {dag_type: [] for dag_type in ['linear', 'branching', 'mixed']}  
    for round in range(args.federated_rounds):
        print(f"Federated Learning Round {round + 1}")
        
        # Training
        for agent, env, dag_type in zip(local_agents, train_envs, ['linear', 'branching', 'mixed']):
            round_reward, _ = train_local(agent, env, episodes=args.local_episodes)
            rewards_train[dag_type].extend(round_reward)
            writer.add_scalar(f'Train Reward/{dag_type}', np.mean(round_reward), round)

        # Validation
        for agent, env, dag_type in zip(local_agents, val_envs, ['linear', 'branching', 'mixed']):
            val_reward = test_agent(agent, env, num_episodes=args.test_episodes)
            rewards_val[dag_type].append(val_reward)
            writer.add_scalar(f'Validation Reward/{dag_type}', val_reward, round)

        # Perform federated averaging
        federated_averaging(global_agent, local_agents)

        # Test global agent on validation environments
        for env, dag_type in zip(val_envs, ['linear', 'branching', 'mixed']):
            global_val_reward = test_agent(global_agent, env, num_episodes=args.test_episodes)
            global_rewards_val[dag_type].append(global_val_reward)
            writer.add_scalar(f'Global Validation Reward/{dag_type}', global_val_reward, round)

    writer.close()

    return rewards_train, rewards_val, global_rewards_val  

def plot_training_validation_rewards(rewards_train, rewards_val, global_rewards_val):
    plt.figure(figsize=(12, 6))
    for dag_type in ['linear', 'branching', 'mixed']:
        plt.plot(rewards_train[dag_type], label=f'{dag_type} Train')
        plt.plot(rewards_val[dag_type], label=f'{dag_type} Validation')
        plt.plot(global_rewards_val[dag_type], label=f'{dag_type} Global Validation', linestyle='--') 
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Training and Validation Rewards for Different DAG Types')
    plt.legend()
    plt.savefig('training_validation_rewards.png')
    plt.close()


def save_rewards_to_file(rewards_train, rewards_val, global_rewards_val):
    for dag_type in ['linear', 'branching', 'mixed']:
        with open(f'rewards_{dag_type}_train.txt', 'w') as f:
            for reward in rewards_train[dag_type]:
                f.write(f"{reward}\n")
        with open(f'rewards_{dag_type}_val.txt', 'w') as f:
            for reward in rewards_val[dag_type]:
                f.write(f"{reward}\n")
        with open(f'global_rewards_{dag_type}_val.txt', 'w') as f:
            for reward in global_rewards_val[dag_type]:
                f.write(f"{reward}\n")

def main():
    args = parse_arguments()
    setup_environment(args.seed)

    rewards_train, rewards_val, global_rewards_val = train_and_test(args)  
    save_rewards_to_file(rewards_train, rewards_val, global_rewards_val)
    plot_training_validation_rewards(rewards_train, rewards_val, global_rewards_val)

if __name__ == "__main__":
    main()    