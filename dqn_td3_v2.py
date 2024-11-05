import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from FedRL_main_train_only import TRAINING_CONFIG

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return self.max_action * (0.9 * T.sigmoid(self.l4(a)) + 0.1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # First Q function
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)
        # Second Q function
        self.l5 = nn.Linear(state_dim + action_dim, 256)
        self.l6 = nn.Linear(256, 256)
        self.l7 = nn.Linear(256, 256)
        self.l8 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = T.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = T.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, next_state, reward, done):
        data = (state, action, next_state, reward, done)

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (
            np.array(batch_states),
            np.array(batch_actions),
            np.array(batch_next_states),
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1)
        )

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = T.FloatTensor(state.reshape(1, -1))
        state = state.to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        state = T.FloatTensor(state).to(device)
        action = T.FloatTensor(action).to(device)
        next_state = T.FloatTensor(next_state).to(device)
        reward = T.FloatTensor(reward).to(device)
        done = T.FloatTensor(done).to(device)

        continuous_action = action[:, 1:].cpu()
        with T.no_grad():
            noise = T.FloatTensor(continuous_action).data.normal_(0, self.policy_noise)
            noise = noise.to(device).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(0.1, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = T.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.discount * target_Q).detach()
        continuous_action = continuous_action.to(device)
        current_Q1, current_Q2 = self.critic(state, continuous_action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item() + (actor_loss.item() if actor_loss is not None else 0)

    def save(self, filename):
        T.save(self.actor.state_dict(), filename + "_actor")
        T.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename):
        self.actor.load_state_dict(T.load(filename + "_actor"))
        self.critic.load_state_dict(T.load(filename + "_critic"))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

class JointAgent:
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim, max_action):
        # Use lower learning rate
        lr = TRAINING_CONFIG["LEARNING_RATE"]
        
        self.dqn = DeepQNetwork(lr=lr, input_dims=[state_dim], 
                               fc1_dims=256, fc2_dims=256, fc3_dims=256, 
                               n_actions=discrete_action_dim)
        self.dqn_target = copy.deepcopy(self.dqn)
        
        self.td3 = TD3(state_dim, continuous_action_dim, max_action)
        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action
        
        # Modified exploration parameters
        self.epsilon = TRAINING_CONFIG["EPSILON_START"]
        self.epsilon_min = TRAINING_CONFIG["EPSILON_END"]
        self.epsilon_decay = TRAINING_CONFIG["EPSILON_DECAY"]
        
        self.training = True
        self.gamma = 0.99
        self.tau = 0.001


    def train(self):
        self.training = True
        self.dqn.train()
        self.td3.actor.train()
        self.td3.critic.train()

    def eval(self):
        self.training = False
        self.dqn.eval()
        self.td3.actor.eval()
        self.td3.critic.eval()

    def select_action(self, state, random_action=False):
        state = T.FloatTensor(state).unsqueeze(0)
        if self.training and (random_action or np.random.random() < self.epsilon):
            discrete_action = np.random.randint(self.dqn.n_actions)
        else:
            state = state.to(device)
            discrete_action = self.dqn.forward(state).argmax().item()
        
        continuous_action = self.td3.select_action(state.cpu().numpy())
        return discrete_action, continuous_action
    
    def add_transition(self, state, discrete_action, continuous_action, next_state, reward, done):
        discrete_action = np.array([discrete_action], dtype=np.float32)
        action = np.concatenate((discrete_action, continuous_action))
        self.replay_buffer.add(state, action, next_state, reward, done)

    def train(self):
        if len(self.replay_buffer.storage) > 1024:
            states, actions, next_states, rewards, dones = self.replay_buffer.sample(256)

            # Separate discrete and continuous actions
            discrete_actions = T.tensor(actions[:, 0], dtype=T.long, device=device)
            continuous_actions = T.tensor(actions[:, 1:], dtype=T.float32, device=device)

            # Train DQN
            q_pred = self.dqn.forward(T.tensor(states, dtype=T.float32, device=device))
            q_target = self.dqn_target.forward(T.tensor(next_states, dtype=T.float32, device=device)).detach()
            dones = T.tensor(dones, dtype=T.float32, device=device)
            rewards = T.tensor(rewards, dtype=T.float32, device=device)
            q_target_val = rewards + (1.0 - dones) * self.gamma * q_target.max(1)[0].unsqueeze(1)

            q_pred = q_pred.gather(1, discrete_actions.unsqueeze(1)).squeeze(1)
            dqn_loss = self.dqn.loss(q_pred, q_target_val.squeeze(1))

            self.dqn.optimizer.zero_grad()
            dqn_loss.backward()
            nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
            self.dqn.optimizer.step()

            # Update target network
            for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Train TD3
            td3_loss = self.td3.train(self.replay_buffer, batch_size=256)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return dqn_loss.item() + td3_loss

        return 0.0  # Return 0.0 if no training occurred

    def get_weights(self):
        return {
            'dqn': self.dqn.state_dict(),
            'dqn_target': self.dqn_target.state_dict(),
            'actor': self.td3.actor.state_dict(),
            'critic': self.td3.critic.state_dict(),
            'critic_target': self.td3.critic_target.state_dict()
        }

    def set_weights(self, weights):
        self.dqn.load_state_dict(weights['dqn'])
        self.dqn_target.load_state_dict(weights['dqn_target'])
        self.td3.actor.load_state_dict(weights['actor'])
        self.td3.critic.load_state_dict(weights['critic'])
        self.td3.critic_target.load_state_dict(weights['critic_target'])