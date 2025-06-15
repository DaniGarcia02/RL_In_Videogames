import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, AtariPreprocessing, FrameStackObservation
from gymnasium import ActionWrapper
from gymnasium.vector import SyncVectorEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import time
import math
import gc

import ale_py

gym.register_envs(ale_py)

# Hyperparameters
NUM_ENVS = 256
TOTAL_STEPS = 50_000_000
RMS = 300_000
LR = 2.5e-4
MBS = 64
NSR = 10_000
DF = 0.99

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mixed precision tools
from torch.amp import autocast, GradScaler

# Dueling + Double DQN network
torch.manual_seed(0)
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            dummy = F.relu(self.conv3(dummy))
            flat_size = dummy.view(1, -1).size(1)
        self.val_fc = nn.Linear(flat_size, 512)
        self.val_out = nn.Linear(512, 1)
        self.adv_fc = nn.Linear(flat_size, 512)
        self.adv_out = nn.Linear(512, num_actions)
    def forward(self, x):
        # x: Tensor [B, C, H, W]
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        flat = x.view(x.size(0), -1)
        v = F.relu(self.val_fc(flat))
        v = self.val_out(v)                 # [B,1]
        a = F.relu(self.adv_fc(flat))
        a = self.adv_out(a)                 # [B,A]
        return v + (a - a.mean(dim=1, keepdim=True))

# Fast replay buffer
torch.manual_seed(0)
class FastReplayBuffer:
    def __init__(self, capacity, shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        C, H, W = shape
        self.states = np.empty((capacity, C, H, W), dtype=np.uint8)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)
    def append(self, s, a, ns, r, d):
        self.states[self.ptr] = s
        self.next_states[self.ptr] = ns
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        s = torch.from_numpy(self.states[idx]).to(self.device, non_blocking=True)
        ns = torch.from_numpy(self.next_states[idx]).to(self.device, non_blocking=True)
        a = torch.from_numpy(self.actions[idx]).to(self.device)
        r = torch.from_numpy(self.rewards[idx]).to(self.device)
        d = torch.from_numpy(self.dones[idx]).to(self.device)
        return s, a, ns, r, d
    def __len__(self):
        return self.size

# Environment constructors

def make_env():
    def _thunk():
        env = gym.make("ALE/SpaceInvaders-v5", obs_type='rgb', frameskip=1)
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, 4)
        return env
    return _thunk


# Moving average for plotting
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Training loop
def train():
    venv = SyncVectorEnv([make_env() for i in range(NUM_ENVS)])
    memory = FastReplayBuffer(RMS, (4,84,84), DEVICE)

    policy = DuelingDQN(4, venv.single_action_space.n).to(DEVICE)
    target = DuelingDQN(4, venv.single_action_space.n).to(DEVICE)
    target.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    scaler = GradScaler()

    rewards_per_episode = []
    epsilon_history      = []
    ep_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    ep_count   = 0

    obs, _ = venv.reset()
    global_step = 0

    start_time = time.time()
    while global_step < TOTAL_STEPS:
        if global_step < RMS // 2:
            eps = 1
        else:
            k = -math.log(0.01/1) / (TOTAL_STEPS - (RMS // 2))
            eps = 1 * math.exp(-k * (global_step - (RMS // 2)))        

        # compute greedy actions
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(DEVICE)
            qvals = policy(obs_t)
            greedy = qvals.argmax(dim=1).cpu().numpy()
        # mask random
        mask = np.random.rand(NUM_ENVS) < eps
        rand_actions = np.random.randint(0, venv.single_action_space.n, size=NUM_ENVS)
        actions = np.where(mask, rand_actions, greedy)

        # step vector env
        next_obs, rewards, terms, truns, infos = venv.step(actions)
        done = terms | truns
        global_step += NUM_ENVS

        # store each transition
        for i in range(NUM_ENVS):
            r = rewards[i] # reward clipping
            memory.append(obs[i], actions[i], next_obs[i], r, float(done[i]))
            ep_rewards[i] += r
            if done[i]:
                rewards_per_episode.append(ep_rewards[i])
                epsilon_history.append(eps)
                ep_rewards[i] = 0.0
                ep_count += 1
                if ep_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Episode: {ep_count:4d} | Reward: {rewards_per_episode[-1]:6.1f} | "
                          f"Epsilon: {eps:.3f} | Steps: {global_step} | Time: {elapsed:.2f}")

        obs = next_obs

        # training update
        if len(memory) >= RMS // 2:
            s, a, ns, r, d = memory.sample(MBS)
            with autocast(device_type='cuda'):
                q_sa = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
                next_actions = policy(ns).argmax(dim=1, keepdim=True)
                next_qs = target(ns).gather(1, next_actions).squeeze(1)
                q_target = r + DF * next_qs * (1.0 - d)
                loss = F.mse_loss(q_sa, q_target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # sync target
        if global_step % NSR < NUM_ENVS:
            target.load_state_dict(policy.state_dict())

    venv.close()

    # Plotting
    episodes = np.arange(len(rewards_per_episode))
    ma = moving_average(rewards_per_episode)

    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(episodes, rewards_per_episode, alpha=0.3, label='Raw Reward')
    plt.plot(np.arange(len(ma)) + 99, ma, color='green', label='100-ep MA')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Convergence')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(122)
    plt.plot(episodes, epsilon_history)
    plt.xlabel('Episode'); plt.ylabel('Epsilon'); plt.title('Epsilon Decay')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('spaceInvaders_duelingDDQN_exponentialDecay_50M_convergence.png', dpi=300)
    plt.close()

    np.savetxt('spaceInvaders_duelingDDQN_exponentialDecay_50M_rewards_per_episode.txt', rewards_per_episode, fmt='%.6f')
    print(f"Saved {len(rewards_per_episode)} rewards to rewards_per_episode.txt")

    torch.save(policy.state_dict(), 'spaceInvaders_duelingDDQN_exponentialDecay_50M_policy_net.pth')
    print("Saved network weights to policy_net.pth")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    gc.disable()
    train()
