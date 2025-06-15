import torch
import torch.nn.functional as F 
from torch import nn 
from collections import namedtuple, deque
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import ResizeObservation, FrameStackObservation, GrayscaleObservation
import ale_py
import pandas as pd

# Define a named tuple for experiences
Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            dummy = F.relu(self.conv3(dummy))
            self.flattened_size = dummy.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        # Convert numpy array to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
        
        # Handle single observation (add batch dimension)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Normalize pixel values
        x = x.float() / 255.0
        
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, *args):
        """Save an experience"""
        self.buffer.append(Experience(*args))
        
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to tensors in one operation
        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(DEVICE)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.array([e.done for e in experiences], dtype=np.float32)).float().to(DEVICE)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

def DeepQLearning(env, episodes, lr, gamma, sync_rate, buffer_size, batch_size):
    # Initialize networks
    policy_net = DQN(4, env.action_space.n).to(DEVICE)
    target_net = DQN(4, env.action_space.n).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(buffer_size)
    
    # Exploration settings
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99999995
    
    # Tracking metrics
    rewards_history = []
    losses = []
    episode_rewards = []
    
    steps_done = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        while True:
            # Select action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            memory.push(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            steps_done += 1

            
            # Train if enough samples
            if len(memory) > batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                
                # Compute current Q values
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Compute target Q values
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0].detach()
                    target_q = rewards + (gamma * next_q * (1 - dones))
                
                # Compute loss and optimize
                loss = F.mse_loss(current_q.squeeze(), target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                episode_loss.append(loss.item())
            
            # Sync target network
            if steps_done % sync_rate == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if done:
                break
        
        epsilon = 0.05 + (1 - 0.05) * (15000 - episode) / 15000

        # Track metrics
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1}/{episodes} | "
              f"Reward: {episode_reward:.1f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Avg Loss: {avg_loss:.4f}")
    
    # Save model and plots
    torch.save(policy_net.state_dict(), "space_invaders_dqn.pt")
    
    # Plot rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(pd.Series(episode_rewards).rolling(100).mean())
    plt.title("Smoothed Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()
    
    return episode_rewards, losses

def test(env, episodes, model_path):
    policy_net = DQN(4, env.action_space.n).to(DEVICE)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()
    
    rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                action = policy_net(state_tensor).argmax().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    avg_reward = np.mean(rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward

import time
def main():
    # Create environment
    env = gym.make('ALE/SpaceInvaders-v5', obs_type="grayscale", render_mode="rgb_array")
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    
    # Hyperparameters
    params = {
        'episodes': 20000,
        'lr': 1e-4,
        'gamma': 0.99,
        'sync_rate': 1000,
        'buffer_size': 50000,
        'batch_size': 64
    }
    
    st = time.time()
    # Train the model
    rewards, losses = DeepQLearning(env, **params)
    et = time.time()
    np.savetxt('Approximate_Methods/DQN/rewards_dqn.txt',rewards)
    np.savetxt('Approximate_Methods/DQN/losses_dqn.txt', losses)

    # Test the model
    test(env, 10, "space_invaders_dqn.pt")
    
    env.close()

if __name__ == "__main__":
    main()