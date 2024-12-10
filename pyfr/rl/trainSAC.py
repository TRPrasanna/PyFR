# Add imports at top
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import os
from pyfr.rl.env import PyFREnvironment
from tqdm import tqdm
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict import TensorDict

class TwinQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs, act):
        sa = torch.cat([obs, act], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * act_dim)  # Mean and log_std
        )
        self.act_dim = act_dim

    def forward(self, obs):
        x = self.net(obs)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

def train_agent(
    mesh_file,
    cfg_file,
    backend_name,
    checkpoint_dir='checkpoints',
    restart_soln=None,
    load_model=None,
    num_episodes=800,
    actions_per_episode=92,
    batch_size=256,
    lr=3e-4,
    alpha_init=0.2,
    hidden_size=256,
    buffer_size=100_000,
    gamma=0.99,
    tau=0.005,
):
    device = torch.device('cpu')
    env = PyFREnvironment(mesh_file, cfg_file, backend_name, restart_soln)
    
    # Get dimensions from environment specs
    obs_dim = env.observation_spec["observation"].shape[0]
    act_dim = 1  # Single action dimension for PyFR control
    act_low = env.action_spec["action"].space.minimum.item()
    act_high = env.action_spec["action"].space.maximum.item()
    
    # Initialize networks
    actor = ActorNetwork(obs_dim, act_dim, hidden_size).to(device)
    critic = TwinQNetwork(obs_dim, act_dim, hidden_size).to(device)
    critic_target = TwinQNetwork(obs_dim, act_dim, hidden_size).to(device)
    
     # Training loop
    best_reward = float('-inf')
    os.makedirs(checkpoint_dir, exist_ok=True)
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        
        for step in range(actions_per_episode):
            with torch.no_grad():
                mean, log_std = actor(state["observation"])
                std = log_std.exp()
                normal = Normal(mean, std)
                x = normal.rsample()
                action = torch.tanh(x)
                action = act_low + (action + 1.0) * (act_high - act_low) / 2.0
                
                # Create action tensordict
                action_td = TensorDict({
                    "action": action.reshape(1)
                }, batch_size=torch.Size([]))
                
            next_state = env.step(action_td)
            
            # Check for crash
            if next_state["terminated"].item():
                print("Simulation crashed, ending episode")
                break
                
            reward = next_state["next", "reward"].item()
            done = next_state["done"].item()
            episode_reward += reward
            
            # Store transition
            if not done:
                replay_buffer.add({
                    "observation": state["observation"],
                    "action": action.reshape(1),
                    "reward": torch.tensor([reward]),
                    "next_observation": next_state["next", "observation"],
                    "done": torch.tensor([done])
                })
            
            # Rest of training loop remains same...
            
            if done:
                break
                
            state = next_state
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'best': f"{best_reward:.2f}",
            'buffer': f"{len(replay_buffer)}/{buffer_size}",
            'alpha': f"{alpha.item():.2e}"
        })
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'alpha': alpha.item(),
                'reward': best_reward,
            }, f"{checkpoint_dir}/best_model.pt")
            
    env.close()
    return actor, critic, alpha