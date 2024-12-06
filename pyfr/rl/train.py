import torch
from torch import nn
from collections import defaultdict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor, TruncatedNormal
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm.auto import tqdm
from pyfr.rl.env import PyFREnvironment
from torchrl.envs.utils import check_env_specs, step_mdp
import os
from tensordict.tensordict import TensorDict
from typing import Dict, Optional
import numpy as np

class ReplayBuffer:
    """Replay buffer implementation using TensorDict."""
    
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
        
    def push(self, tensordict: TensorDict):
        """Add a new experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = tensordict.clone()
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Optional[TensorDict]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return None
        samples = random.sample(self.buffer, batch_size)
        return torch.stack(samples, dim=0).to(self.device)
        
    def __len__(self):
        return len(self.buffer)

def train_agent(mesh_file, cfg_file, backend_name, checkpoint_dir='checkpoints', restart_soln=None):
    # Device setup
    #device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
    device = torch.device('cpu')
                          
    # Hyperparameters
    num_cells = 512  # Hidden layer size
    num_cells_critic = 32  # Hidden layer size for critic
    frames_per_batch = 80  # 80 actions per batch (and per episode?)
    total_frames = frames_per_batch * 400 # Total number of actions across all episodes (400 episodes)
    num_epochs = 25         # optimization steps per batch?
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.97 #0.95
    entropy_eps = 0.01 #1e-4 
    lr = 1e-3 #3e-4
    max_grad_norm = 1.0

    # Add replay buffer parameters
    buffer_size = 10000
    batch_size = 80
    min_buffer_size = 800  # Min experiences before training

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size, device)

    # Initialize environment
    env = PyFREnvironment(mesh_file, cfg_file, backend_name, restart_soln)
    #check_env_specs(env)
    
     # Actor network with proper output handling
    actor_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], num_cells),
        nn.Tanh(),
        nn.Linear(num_cells, num_cells),
        nn.Tanh(),
        nn.Linear(num_cells, 2),  # 2 outputs: mean and log_std
        NormalParamExtractor()  # Use default scale_mapping
    ).to(device)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"]  # NormalParamExtractor splits into these
    ).to(device)
    
    policy = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        #safe = True
    ).to(device)

    # Value network (critic)
    value_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], num_cells_critic),
        nn.Tanh(),
        nn.Linear(num_cells_critic, num_cells_critic),
        nn.Tanh(),
        nn.Linear(num_cells_critic, 1)
    ).to(device)

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"]
    ).to(device)

    # PPO components
    advantage_module = GAE(
        gamma=gamma, 
        lmbda=lmbda,
        value_network=value_module,
        average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
    )

    # Optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
    
    # Data collection
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device
    )

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_reward = float('-inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')

    # Training loop
    total_episodes = total_frames // frames_per_batch
    pbar = tqdm(total=total_episodes, desc="Training")
    episode_count = 0

    for i, tensordict_data in enumerate(collector):
        # Add experiences to replay buffer
        replay_buffer.push(tensordict_data)
        
        # Only train if we have enough samples
        if len(replay_buffer) >= min_buffer_size:
            for _ in range(num_epochs):
                # Sample from replay buffer
                batch = replay_buffer.sample(batch_size)
                if batch is None:
                    continue
                    
                # Calculate advantages
                advantage_module(batch)
                data_view = batch.reshape(-1)
                
                # Training step
                loss_vals = loss_module(data_view)
                loss_value = (
                    loss_vals["loss_objective"] + 
                    loss_vals["loss_critic"] + 
                    loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        # Get episode reward
        mean_reward = tensordict_data["next", "reward"].mean().item()
        #logs["reward"].append(mean_reward)

        # Save checkpoint periodically
        # if episode_count % 10 == 0:
        #     checkpoint = {
        #         'policy_state_dict': policy.state_dict(),
        #         'value_state_dict': value_module.state_dict(),
        #         'optimizer_state_dict': optim.state_dict(),
        #         'episode': episode_count,
        #         'mean_reward': mean_reward,
        #     }
        #     torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_{episode_count}.pt")
            
        # Update episode count and progress bar
        episode_count += 1
        pbar.set_postfix({
            "reward": f"{mean_reward:.2f}",
            "best": f"{best_reward:.2f}"
        })
        pbar.update(1)
        
        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            print(f"\nNew best reward: {best_reward:.4f}")
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_module.state_dict(),
                'reward': best_reward,
                'episode': episode_count,
            }, best_model_path)

    collector.shutdown()
    env.close()
    #return logs
