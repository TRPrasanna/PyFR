from pyfr.inifile import Inifile
from dataclasses import dataclass
from pyfr.readers.native import NativeReader
from typing import Dict, Any
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
import math
import numpy as np

def train_agent(mesh_file, cfg_file, backend_name, checkpoint_dir='checkpoints', ic_dir=None, load_model=None):
    # Device setup
    #device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
    #device = torch.device('cpu')
    device = torch.device('cuda')

    cfg = Inifile.load(cfg_file)
    mesh = NativeReader(mesh_file)
    if 'neuralnetwork-hyperparameters' not in cfg.sections():
        print("No neuralnetwork-hyperparameters section found in config file. Proceeding to use default hyperparameters.")

    # Initialize environment
    env = PyFREnvironment(mesh, cfg, backend_name, ic_dir=ic_dir)
    env = TransformedEnv(env,StepCounter())

    hp = HyperParameters.from_config(cfg)
    # Calculate derived parameters using environment info
    hp._calculate_derived(env)

    # Adjust num_minibatches if it does not divide frames_per_batch evenly
    sub_batch_size = hp.frames_per_batch // hp.desired_num_minibatches
    remainder = hp.frames_per_batch % hp.desired_num_minibatches
    if remainder != 0:
        adjusted_num_minibatches = get_closest_divisor(hp.frames_per_batch, hp.desired_num_minibatches)
        sub_batch_size = hp.frames_per_batch // adjusted_num_minibatches
        print(
            f"Warning: frames_per_batch ({hp.frames_per_batch}) is not perfectly divisible by "
            f"num_minibatches ({hp.desired_num_minibatches}). "
            f"Adjusted num_minibatches to {adjusted_num_minibatches} with sub_batch_size {sub_batch_size}."
        )
        hp.desired_num_minibatches = adjusted_num_minibatches

    hp.print_summary()

     # Actor network with proper output handling
    actor_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], hp.num_cells_policy),
        nn.Tanh(), # tanh activation function is most commonly used for small networks for PPO
        nn.Linear(hp.num_cells_policy, hp.num_cells_policy),
        nn.Tanh(),
        nn.Linear(hp.num_cells_policy, 2),  # 2 outputs: mean and log_std
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
        nn.Linear(env.observation_spec["observation"].shape[0], hp.num_cells_value),
        nn.Tanh(),
        nn.Linear(hp.num_cells_value, hp.num_cells_value),
        nn.Tanh(),
        nn.Linear(hp.num_cells_value, 1)
    ).to(device)

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"]
    ).to(device)

    # PPO components
    advantage_module = GAE(
        gamma=hp.gamma, 
        lmbda=hp.lmbda,
        value_network=value_module,
        average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=hp.clip_epsilon,
        entropy_bonus=bool(hp.entropy_eps),
        entropy_coef=hp.entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # Optimizer
    optim = torch.optim.Adam(loss_module.parameters(), hp.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, hp.total_frames // hp.frames_per_batch, 0.0
    )

    # Data collection
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=hp.frames_per_batch,
        total_frames=hp.total_frames,
        split_trajs=False,
        reset_at_each_iter=True, # without this the collector seems to continue collecting in evaluation mode
        device=device
    )

    # Replay buffer here is not actually used for experience replay
    # It is rather used for convenience to sample mini-batches from the collected data
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=hp.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    best_eval_reward = float('-inf')
    best_eval_episode = 0
    start_episode = 0
    current_eval_reward = None

    # Load existing model if specified
    if load_model and os.path.exists(load_model):
        checkpoint = torch.load(load_model, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        value_module.load_state_dict(checkpoint['value_state_dict'])
        
        # Handle both old and new checkpoint formats
        current_eval_reward = checkpoint.get('current_reward', checkpoint.get('reward', None))
        loaded_best_reward = checkpoint.get('best_reward', current_eval_reward)
        start_episode = checkpoint.get('episode', 0)
        loaded_best_episode = checkpoint.get('best_episode', start_episode)

        print("\nLoaded existing model:")
        if current_eval_reward is not None:
            print(f"Current eval reward: {current_eval_reward:.4f}")
        if loaded_best_reward is not None:
            print(f"Best eval reward: {loaded_best_reward:.4f}")
            print(f"Best reward achieved at episode: {loaded_best_episode}")
        print(f"Current episode count: {start_episode}")
        print(f"Model path: {load_model}\n")

        # Only update best reward if loading the best model
        if "best_model" in load_model:
            best_eval_reward = loaded_best_reward
            best_eval_episode = loaded_best_episode
        else:
            print("Note: Loading non-best model, will track new best reward from here\n")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    latest_model_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    logs = defaultdict(list)
    remaining_episodes = hp.episodes - start_episode
    pbar = tqdm(total=remaining_episodes, desc="Training", initial=start_episode)
    episode_count = start_episode

    eval_str = ""

    for i, tensordict_data in enumerate(collector):
        print_tensordict_diagnostics(tensordict_data, verbose=True)
        # Training updates
        for _ in range(hp.num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            
            for _ in range(hp.frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"]
                if hp.entropy_eps > 0:
                    loss_value = loss_value + loss_vals["loss_entropy"]

                loss_value.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), hp.max_grad_norm)
                optim.step()
                optim.zero_grad()

        # Logging
        episode_count += hp.episodes_per_batch
        train_reward = tensordict_data["next", "reward"].mean().item()
        logs["train_reward"].append(train_reward)
        #print(f"\n Batch finished. Episode count is {episode_count}")

        # Evaluate every hp.eval_frequency batches
        if i % hp.eval_frequency == 0:
            eval_reward = evaluate_policy(env, policy)
            logs["eval_reward"].append(eval_reward)
            
            # Save best model if new best achieved
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_episode = episode_count
                print(f"\nNew best eval reward: {best_eval_reward:.5f} at episode {episode_count}")
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'value_state_dict': value_module.state_dict(),
                    'current_reward': eval_reward,
                    'best_reward': best_eval_reward,
                    'episode': episode_count,
                    'best_episode': best_eval_episode,
                }, best_model_path)

            # Save latest model
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_module.state_dict(),
                'current_reward': eval_reward,
                'best_reward': best_eval_reward,
                'episode': episode_count,
                'best_episode': best_eval_episode,
            }, latest_model_path)

            eval_str = f"eval reward: {eval_reward:.5f} (best: {best_eval_reward:.5f})"

        # Progress bar update
        pbar.set_postfix({
            "train_reward": f"{train_reward:.5f}",
            "eval": eval_str,
            "lr": f"{optim.param_groups[0]['lr']:.2e}",
        })
        pbar.update(hp.episodes_per_batch)

        scheduler.step()

    collector.shutdown()
    env.close()

def evaluate_policy(env, policy, num_steps=1000000): 
    # _check_done will take care of num_steps, but done is not resetting env for some reason
    """Evaluate policy without exploration using consistent IC"""
    #print("Evaluating policy...")
    env.set_evaluation_mode(True)  # Use same IC
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(num_steps, policy)
            eval_reward = eval_rollout["next", "reward"].mean().item()
            del eval_rollout
            return eval_reward
    finally:
        env.set_evaluation_mode(False)  # Reset to training mode
        #print("Evaluation complete.Returning to training mode.")

def get_closest_divisor(n, target):
    """
    Finds the closest divisor of n to the target value.
    
    Args:
        n (int): The number to find divisors for.
        target (int): The target divisor to approach.
        
    Returns:
        int: The closest divisor to the target.
    """
    # Find all divisors of n
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    
    # Find the divisor with the minimum absolute difference to the target
    closest = min(divisors, key=lambda x: (abs(x - target), -x))  # Prefer larger divisor if tie
    return closest

@dataclass
class HyperParameters:
    # Network architecture
    num_cells_policy: int = 512
    num_cells_value: int = 512
    
    # Training schedule
    episodes: int = 1200
    episodes_per_batch: int = 20
    desired_num_minibatches: int = 16
    num_epochs: int = 10
    
    # PPO parameters
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.97
    entropy_eps: float = 1e-3
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    # Evaluation settings
    eval_frequency: int = 1  # Evaluate every N policy updates

    def __post_init__(self):
        """Initialize parameter sources and calculate derived values"""
        # Mark all parameters as default initially
        self._param_sources = {
            field_name: 'default' 
            for field_name in self.__dataclass_fields__.keys()
        }
        self._derived_params = {'frames_per_batch', 'total_frames', 'actions_per_episode'}
        self.actions_per_episode = None
        self.frames_per_batch = None
        self.total_frames = None

    def _calculate_derived(self, env):
        """Calculate derived parameters"""
        self.actions_per_episode = int(env.dtend / env.action_interval)
        self.frames_per_batch = self.episodes_per_batch * self.actions_per_episode
        self.total_frames = self.episodes * self.actions_per_episode

    @classmethod
    def from_config(cls, cfg: Inifile) -> 'HyperParameters':
        params = cls()
        if 'neuralnetwork-hyperparameters' in cfg.sections():
            section = 'neuralnetwork-hyperparameters'
            for field_name, field in params.__dataclass_fields__.items():
                # Convert underscore to hyphen for config lookup
                config_name = field_name.replace('_', '-')
                if field_name not in params._derived_params and cfg.hasopt(section, config_name):
                    if field.type == int:
                        value = cfg.getint(section, config_name)
                    elif field.type == float:
                        value = cfg.getfloat(section, config_name)
                    elif field.type == bool:
                        value = cfg.getbool(section, config_name)
                    else:
                        value = cfg.get(section, config_name)
                    setattr(params, field_name, value)
                    params._param_sources[field_name] = 'config'
        return params

    def print_summary(self) -> None:
        """Print hyperparameter summary with sections and sources"""
        sections = {
            "Network Architecture": [
                ("num_cells_policy", "Size of policy network hidden layers"),
                ("num_cells_value", "Size of value network hidden layers")
            ],
            "Training Schedule": [
                ("episodes", "Total training episodes"),
                ("episodes_per_batch", "Episodes per update batch"),
                ("desired_num_minibatches", "Target minibatches per update"),
                ("num_epochs", "Training epochs per batch")
            ],
            "PPO Parameters": [
                ("clip_epsilon", "PPO clipping parameter"),
                ("gamma", "Discount factor"),
                ("lmbda", "GAE lambda parameter"),
                ("entropy_eps", "Entropy bonus coefficient"),
                ("lr", "Learning rate"),
                ("max_grad_norm", "Gradient clipping norm")
            ],
            "Derived Values": [
                ("frames_per_batch", "Frames per batch"),
                ("total_frames", "Total training frames"),
                ("actions_per_episode", "Actions per episode")
            ],
            "Evaluation Settings": [
                ("eval_frequency", "Evaluate policy every N updates"),
            ]
        }

        def format_line(param: str, value: Any, desc: str, source: str) -> str:
            if param in self._derived_params:
                src_mark = "[-]"
            else:
                src_mark = "[C]" if source == "config" else "[D]"
            return f"| {param:<25} | {str(value):<15} | {src_mark:<5} | {desc:<40} |"

        def print_header():
            return (f"| {'Parameter':<25} | {'Value':<15} | {'Src':<5} | {'Description':<40} |\n" + 
                   f"|{'-'*27}|{'-'*17}|{'-'*7}|{'-'*42}|")

        print("\nHyperparameters Configuration")
        print("=" * 98)

        for section_name, params in sections.items():
            print(f"\n{section_name}:")
            print("=" * 98)
            print(print_header())
            
            for param_name, description in params:
                value = getattr(self, param_name)
                source = self._param_sources.get(param_name, 'derived')
                print(format_line(param_name, value, description, source))
            
            print("-" * 98)

        # Print legend
        print("\nSource: [C]=From .ini config file, [D]=Default, [-]=Derived")

def print_tensordict_diagnostics(td, verbose=True):
    """Print human-readable diagnostics of tensordict data."""
    print("\n=== TensorDict Diagnostics ===")
    
    batch_size = td.batch_size[0]
    print(f"\nBatch contains {batch_size} transitions")
    
    
    print(f"Next Done: {td['next','done']}")
    if verbose:
        print("\nTransition Details:")
        print("-" * 80)
        for i in range(batch_size):
            print(f"\nStep {i}:")
            print(f"Observation shape: {td['observation'][i].shape}")
            print(f"Observation: {td['observation'][i].cpu().numpy()}")
            #next_observation = td['next', 'observation'][i].cpu().numpy()
            #print(f"Next Observation: {next_observation}")
            
            # Handle multi-dimensional actions
            action = td['action'][i].cpu().numpy()
            if len(action.shape) > 0:
                action_str = ", ".join([f"{a:.4f}" for a in action])
            else:
                action_str = f"{action:.4f}"
            print(f"Action: [{action_str}]")
            
            print(f"Reward: {td['next', 'reward'][i].cpu().item():.4f}")
            print(f"Step count: {td['step_count'][i].cpu().item()}")
            print(f"Done: {td['done'][i].cpu().item()}")
            print(f"Terminated: {td['terminated'][i].cpu().item()}")
            print(f"Truncated: {td['truncated'][i].cpu().item()}")
            print(f"Next Done: {td['next','done'][i].cpu().item()}")
            print(f"Next Terminated: {td['next','terminated'][i].cpu().item()}")
            print(f"Next Truncated: {td['next','truncated'][i].cpu().item()}")
        
    
    print("\n" + "="*80)