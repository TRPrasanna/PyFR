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
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import os

def train_agent(mesh_file, cfg_file, backend_name, checkpoint_dir='checkpoints', restart_soln=None, load_model=None):
    # Device setup
    #device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
    #device = torch.device('cpu')
    device = torch.device('cuda')

    # Hyperparameters
    num_cells_policy = 512  # Hidden layer size for policy network
    num_cells_value = 32  # Hidden layer size for value network
    episodes = 1200
    actions_per_episode = 480 # 93?
    episodes_per_batch = 20 #1
    frames_per_batch = actions_per_episode * episodes_per_batch
    total_frames = actions_per_episode * episodes  # 93 actions per episode, 400 episodes
    sub_batch_size = frames_per_batch #round(0.2 * frames_per_batch) # 20% of frames per batch
    num_epochs = 50 #25         # optimization steps per batch
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.97 #0.95
    entropy_eps = 0.01 #1e-3 and 1e-4 seems to crash
    lr = 1e-4 #1e-3 #3e-4
    max_grad_norm = 1.0

    # Initialize environment
    env = PyFREnvironment(mesh_file, cfg_file, backend_name, restart_soln)
    #check_env_specs(env)
    #env = TransformedEnv(
    #    base_env,
    #    Compose(
    #        # normalize observations
    #        ObservationNorm(in_keys=["observation"]),
    #        #DoubleToFloat(), # will probably need this when using PyFR in double precision mode
    #        StepCounter(),
    #    ),
    #)

    #env.transform[0].init_stats(num_iter=actions_per_episode, reduce_dim=0, cat_dim=0)
    #print("Finished gathering stats for observation normalization")

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

    best_eval_reward = float('-inf')
    # Load existing model if specified
    best_reward = float('-inf')
    if load_model and os.path.exists(load_model):
        checkpoint = torch.load(load_model, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        value_module.load_state_dict(checkpoint['value_state_dict'])
        best_eval_reward = checkpoint['reward']
        start_episode = checkpoint.get('episode', 0)
        
        print("\nLoaded existing model:")
        print(f"Previous best eval reward: {best_eval_reward:.4f}")
        print(f"Previous episode count: {start_episode}")
        print(f"Model path: {load_model}\n")
    else:
        start_episode = 0

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    latest_model_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    logs = defaultdict(list)
    remaining_episodes = episodes - start_episode
    pbar = tqdm(total=remaining_episodes, desc="Training", initial=start_episode)
    episode_count = start_episode

    eval_str = ""

    for i, tensordict_data in enumerate(collector):
        # Training updates
        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"] + 
                    loss_vals["loss_critic"] + 
                    loss_vals["loss_entropy"]
                )

                loss_value.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        # Logging
        episode_count += episodes_per_batch
        train_reward = tensordict_data["next", "reward"].mean().item()
        logs["train_reward"].append(train_reward)
        
        # Evaluate every 10 episodes
        if episode_count % 10 < episodes_per_batch:
            eval_reward = evaluate_policy(env, policy, num_steps=actions_per_episode)
            logs["eval_reward"].append(eval_reward)
            
            # Save best
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f"\nNew best eval reward: {best_eval_reward:.4f}")
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'value_state_dict': value_module.state_dict(),
                    'reward': best_eval_reward,
                    'episode': episode_count,
                }, best_model_path)

            # Save latest model even if not the best
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_module.state_dict(),
                'reward': eval_reward,
                'episode': episode_count,
            }, latest_model_path)

            eval_str = f"eval reward: {eval_reward:.2f} (best: {best_eval_reward:.2f})"

        # Progress bar update
        pbar.set_postfix({
            "train_reward": f"{train_reward:.2f}",
            "eval": eval_str,
            "lr": f"{optim.param_groups[0]['lr']:.2e}",
        })
        pbar.update(episodes_per_batch)

        scheduler.step()

    collector.shutdown()
    env.close()

def evaluate_policy(env, policy, num_steps=93):
    """Evaluate policy without exploration"""
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        eval_rollout = env.rollout(num_steps, policy)
        eval_reward = eval_rollout["next", "reward"].mean().item()
        return eval_reward
