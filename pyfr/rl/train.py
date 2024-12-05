import torch
from torch import nn
from collections import defaultdict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor, TruncatedNormal
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm
from pyfr.rl.env import PyFREnvironment
from torchrl.envs.utils import check_env_specs, step_mdp
import os

def train_agent(mesh_file, cfg_file, backend_name, checkpoint_dir='checkpoints'):
    # Device setup
    device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
    
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

    # Initialize environment
    env = PyFREnvironment(mesh_file, cfg_file, backend_name)
    #check_env_specs(env)
    #td = env.rand_step()
    #print("random step tensordict", td)
    #rollout = env.rollout(3)
    #print("rollout of three steps:", rollout)
    #print("Shape of the rollout TensorDict:", rollout.batch_size)
    
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

    class ScaledTanhNormal(TanhNormal): # this is probably wrong because of log probabilities
        def __init__(self, loc, scale):
            super().__init__(loc, scale)
            self.output_scale = 0.06  # Scale factor to get [-0.1, 0.1]
        
        def rsample(self, sample_shape=torch.Size()):
            x = super().rsample(sample_shape)
            return x * self.output_scale
    
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
    logs = defaultdict(list)
    total_episodes = total_frames // frames_per_batch
    pbar = tqdm(total=total_episodes, desc="Training Progress")
    episode_count = 0

    for i, tensordict_data in enumerate(collector):
        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            
            # Single mini-batch per update
            mini_batch = data_view
            loss_vals = loss_module(mini_batch)
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
        logs["reward"].append(mean_reward)

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

        pbar.set_description(
            f"Episode {episode_count}/{total_episodes} | "
            f"Reward: {mean_reward:.4f} (Best: {best_reward:.4f})"
        )

    collector.shutdown()
    env.close()
    return logs
