import torch
from torch import nn
from collections import defaultdict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm.auto import tqdm
from pyfr.rl.env import PyFREnvironment
#from torchrl.envs.utils import check_env_specs
import os

def train_agent(mesh_file, cfg_file, backend_name, checkpoint_dir='checkpoints', restart_soln=None):
    # Device setup
    #device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
    device = torch.device('cpu')
                          
    # Hyperparameters
    num_cells_policy = 512  # Hidden layer size
    num_cells_value = 32  # Hidden layer size for value network
    episodes = 800
    actions_per_episode = 93
    episodes_per_batch = 1
    frames_per_batch = actions_per_episode * episodes_per_batch
    total_frames = actions_per_episode * episodes  # 93 actions per episode, 400 episodes
    sub_batch_size = round(0.2 * frames_per_batch) # 20% of frames per batch
    num_epochs = 25         # optimization steps per batch
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.97 #0.95
    entropy_eps = 0.01 #1e-4 
    lr = 1e-3 #3e-4
    max_grad_norm = 1.0

    # Initialize environment
    base_env = PyFREnvironment(mesh_file, cfg_file, backend_name, restart_soln)
    #check_env_specs(env)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            #DoubleToFloat(), # will need this when using PyFR in double precision mode
            StepCounter(),
        ),
    )
    
    env.transform[0].init_stats(num_iter=actions_per_episode, reduce_dim=0, cat_dim=0)
    print("Finished gathering stats for observation normalization")

     # Actor network with proper output handling
    actor_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], num_cells_policy),
        nn.Tanh(), # tanh activation function is most commonly used for small networks for PPO
        nn.Linear(num_cells_policy, num_cells_policy),
        nn.Tanh(),
        nn.Linear(num_cells_policy, 2),  # 2 outputs: mean and log_std
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
        distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
        },
        #safe = True
    ).to(device)

    # Value network (critic)
    value_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], num_cells_value),
        nn.Tanh(),
        nn.Linear(num_cells_value, num_cells_value),
        nn.Tanh(),
        nn.Linear(num_cells_value, 1)
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
        loss_critic_type="smooth_l1",
    )

    # Optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
    )
    
    # Data collection
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_reward = float('-inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    logs = defaultdict(list)
    pbar = tqdm(total=episodes, desc="Training")
    episode_count = 0

    # Training loop
    for tensordict_data in collector:
        # PPO update loop
        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            
            # Sub-batch updates
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"] + 
                    loss_vals["loss_critic"] + 
                    loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        # Logging
        mean_reward = tensordict_data["next", "reward"].mean().item()
        logs["reward"].append(mean_reward)
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        logs["lr"].append(optim.param_groups[0]["lr"])

        # Progress updates
        episode_count += episodes_per_batch
        pbar.set_postfix({
            "reward": f"{mean_reward:.2f}",
            "best": f"{best_reward:.2f}",
            "lr": f"{logs['lr'][-1]:.2e}"
        })
        pbar.update(episodes_per_batch)  # Update per batch

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

        # Update learning rate
        scheduler.step()

    collector.shutdown()
    env.close()