import os
import dqn_td3_v2
import torch as T
import json
import glob
import numpy as np


device = T.device("cuda" if T.cuda.is_available() else "cpu")

def save_model(agent, filepath):
    """
    Save a model's state to disk.
    
    Args:
        agent: The agent instance to save
        filepath: Path where to save the model
    """
    save_dict = {
        'dqn_state': agent.dqn.state_dict(),
        'dqn_target_state': agent.dqn_target.state_dict(),
        'td3_actor_state': agent.td3.actor.state_dict(),
        'td3_critic_state': agent.td3.critic.state_dict(),
        'td3_actor_target_state': agent.td3.actor_target.state_dict(),
        'td3_critic_target_state': agent.td3.critic_target.state_dict(),
        'epsilon': agent.epsilon,
        'training_metrics': agent.training_metrics if hasattr(agent, 'training_metrics') else None
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    try:
        T.save(save_dict, filepath)
        print(f"Model successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving model to {filepath}: {str(e)}")

def load_model(agent, filepath):
    """
    Load a model's state from disk.
    
    Args:
        agent: The agent instance to load into
        filepath: Path from where to load the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
            
        checkpoint = T.load(filepath, map_location=device)
        
        # Load DQN states
        agent.dqn.load_state_dict(checkpoint['dqn_state'])
        agent.dqn_target.load_state_dict(checkpoint['dqn_target_state'])
        
        # Load TD3 states
        agent.td3.actor.load_state_dict(checkpoint['td3_actor_state'])
        agent.td3.critic.load_state_dict(checkpoint['td3_critic_state'])
        agent.td3.actor_target.load_state_dict(checkpoint['td3_actor_target_state'])
        agent.td3.critic_target.load_state_dict(checkpoint['td3_critic_target_state'])
        
        # Load training state
        agent.epsilon = checkpoint['epsilon']
        if 'training_metrics' in checkpoint and hasattr(agent, 'training_metrics'):
            agent.training_metrics = checkpoint['training_metrics']
            
        print(f"Model successfully loaded from {filepath}")
        return True
        
    except Exception as e:
        print(f"Error loading model from {filepath}: {str(e)}")
        return False

def save_checkpoint(global_agent, local_agents, metrics, round_idx, args):
    """
    Save a complete training checkpoint including global and local agents.
    
    Args:
        global_agent: The global federated agent
        local_agents: Dictionary of local agents
        metrics: Dictionary of training metrics
        round_idx: Current federated round index
        args: Training arguments
    """
    checkpoint_dir = os.path.join(args.model_dir, f'checkpoint_round_{round_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save global agent
    save_model(global_agent, os.path.join(checkpoint_dir, 'global_agent.pt'))
    
    # Save local agents
    for dag_type, agent in local_agents.items():
        save_model(agent, os.path.join(checkpoint_dir, f'local_agent_{dag_type}.pt'))
    
    # Save metrics and arguments
    metrics_file = os.path.join(checkpoint_dir, 'metrics.json')
    args_file = os.path.join(checkpoint_dir, 'args.json')
    
    try:
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Save arguments
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, indent=4)
            
        print(f"Checkpoint saved at {checkpoint_dir}")
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")

def load_checkpoint(global_agent, local_agents, checkpoint_dir):
    """
    Load a complete training checkpoint.
    
    Args:
        global_agent: The global federated agent to load into
        local_agents: Dictionary of local agents to load into
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        tuple: (metrics, args) if successful, (None, None) otherwise
    """
    try:
        # Load global agent
        global_model_path = os.path.join(checkpoint_dir, 'global_agent.pt')
        if not load_model(global_agent, global_model_path):
            return None, None
            
        # Load local agents
        for dag_type, agent in local_agents.items():
            local_model_path = os.path.join(checkpoint_dir, f'local_agent_{dag_type}.pt')
            if not load_model(agent, local_model_path):
                return None, None
        
        # Load metrics and arguments
        metrics_file = os.path.join(checkpoint_dir, 'metrics.json')
        args_file = os.path.join(checkpoint_dir, 'args.json')
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        with open(args_file, 'r') as f:
            args = json.load(f)
            
        print(f"Checkpoint loaded from {checkpoint_dir}")
        return metrics, args
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None, None

def resume_training(checkpoint_dir, state_dim, discrete_action_dim, continuous_action_dim, max_action):
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        state_dim: State dimension
        discrete_action_dim: Discrete action dimension
        continuous_action_dim: Continuous action dimension
        max_action: Maximum action value
        
    Returns:
        tuple: (global_agent, local_agents, metrics, args) if successful, (None, None, None, None) otherwise
    """
    try:
        # Initialize agents
        global_agent = dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)
        
        # Get DAG types from checkpoint directory
        local_agent_files = glob.glob(os.path.join(checkpoint_dir, 'local_agent_*.pt'))
        dag_types = [os.path.splitext(f)[0].split('_')[-1] for f in local_agent_files]
        
        local_agents = {
            dag_type: dqn_td3_v2.JointAgent(state_dim, discrete_action_dim, continuous_action_dim, max_action)
            for dag_type in dag_types
        }
        
        # Load checkpoint
        metrics, args = load_checkpoint(global_agent, local_agents, checkpoint_dir)
        if metrics is None or args is None:
            return None, None, None, None
            
        return global_agent, local_agents, metrics, args
        
    except Exception as e:
        print(f"Error resuming training: {str(e)}")
        return None, None, None, None