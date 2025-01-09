from pyfr.inifile import Inifile
from dataclasses import dataclass
from pyfr.readers.native import NativeReader
from typing import Dict, Any
import torch
from torch import nn
from collections import defaultdict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from tqdm.auto import tqdm
from pyfr.rl.env import PyFREnvironment
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import os
import math

def train_agent(mesh_file, cfg_file, backend_name, checkpoint_dir='checkpoints', restart_soln=None, load_model=None):
    # Device setup
    #device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
    #device = torch.device('cpu')
    device = torch.device('cuda')

    cfg = Inifile.load(cfg_file)
    mesh = NativeReader(mesh_file)
    if 'neuralnetwork-hyperparameters' not in cfg.sections():
        print("No neuralnetwork-hyperparameters section found in config file. Proceeding to use default hyperparameters.")

    hp = HyperParameters.from_config(cfg)
    
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

    # Initialize environment
    env = PyFREnvironment(mesh, cfg, backend_name, restart_soln)

     # Actor network with proper output handling
    actor_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], hp.num_cells_policy),
        nn.ReLU(),
        nn.Linear(hp.num_cells_policy, hp.num_cells_policy),
        nn.ReLU(),
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
        return_log_prob=False,
        distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
        },
        #safe = True
    ).to(device)

    # state-action value (Q) network
    # We won't use the third optional state value network
    class QValueNet(nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, observation, action):
            x = torch.cat([observation, action], dim=-1)
            return self.net(x)

    # Initialize Q-networks
    qvalue_net = QValueNet(
        obs_dim=env.observation_spec["observation"].shape[0],
        action_dim=env.action_spec.shape[0],
        hidden_dim=hp.num_cells_value
    ).to(device)

    qvalue_module = ValueOperator(
        module=qvalue_net,
        in_keys=["observation", "action"],
        out_keys=["state_action_value"]
    ).to(device)

    loss_module = SACLoss(
        actor_network=policy,
        qvalue_network=qvalue_module, #[qvalue_module,qvalue_module2], # here specified as a list so it will be convenient to save as pytorch model
        #num_qvalue_nets=2, 2 is default
        #loss_function='l2',
        #delay_actor=False,
        #delay_qvalue=True,
        #alpha_init=1.0
    )
    loss_module.make_value_estimator(gamma=hp.gamma)
    target_update_polyak = 0.995 # check what this is
    target_net_updater = SoftUpdate(loss_module, eps=target_update_polyak)

    # Optimizer
    optim = torch.optim.Adam(loss_module.parameters(), hp.lr)

    # Data collection
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=hp.frames_per_batch,
        total_frames=hp.total_frames,
        split_trajs=False,
        device=device
    )

    buffer_size = 1000000
    batch_size = 256 # this is the batch size of replay buffer sample
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=LazyTensorStorage(max_size=buffer_size),
        batch_size=batch_size,
    )

    best_eval_reward = float('-inf')
    # Load existing model if specified
    best_reward = float('-inf')
    if load_model and os.path.exists(load_model):
        checkpoint = torch.load(load_model, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        qvalue_module.load_state_dict(checkpoint['qvalue_state_dict'][0])
        #qvalue_module2.load_state_dict(checkpoint['qvalue_state_dict'][1])
        best_eval_reward = checkpoint['best_reward']
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
    remaining_episodes = hp.episodes - start_episode
    pbar = tqdm(total=remaining_episodes, desc="Training", initial=start_episode)
    episode_count = start_episode

    eval_str = ""

    utd_ratio = 1.0 # update to data ratio
    for i, tensordict_data in enumerate(collector):
        collector.update_policy_weights_()
        # Training updates
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        
        for _ in range(int(hp.frames_per_batch * utd_ratio)):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_actor"] + 
                loss_vals["loss_qvalue"] + 
                loss_vals["loss_alpha"]
            )

            loss_value.backward()
            #nn.utils.clip_grad_norm_(loss_module.parameters(), hp.max_grad_norm)
            optim.step()
            optim.zero_grad(set_to_none=True)
            target_net_updater.step()

        # Logging
        episode_count += hp.episodes_per_batch
        train_reward = tensordict_data["next", "reward"].mean().item()
        logs["train_reward"].append(train_reward)
        
        # Evaluate every 10 episodes
        if episode_count % 10 < hp.episodes_per_batch:
            eval_reward = evaluate_policy(env, policy, num_steps=hp.actions_per_episode)
            logs["eval_reward"].append(eval_reward)
            
            # Save best
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f"\nNew best eval reward: {best_eval_reward:.4f}")
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'qvalue_state_dict': [qvalue_module.state_dict(), qvalue_module.state_dict()],
                    'best_reward': best_eval_reward,
                    'latest_reward': eval_reward,
                    'episode': episode_count,
                }, best_model_path)

            # Save latest model even if not the best
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'qvalue_state_dict': [qvalue_module.state_dict(), qvalue_module.state_dict()],
                'best_reward': best_eval_reward,
                'latest_reward': eval_reward,
                'episode': episode_count,
            }, latest_model_path)

            eval_str = f"eval reward: {eval_reward:.2f} (best: {best_eval_reward:.2f})"

        # Progress bar update
        pbar.set_postfix({
            "train_reward": f"{train_reward:.2f}",
            "eval": eval_str,
            "lr": f"{optim.param_groups[0]['lr']:.2e}",
        })
        pbar.update(hp.episodes_per_batch)

    collector.shutdown()
    env.close()

def evaluate_policy(env, policy, num_steps=1000000):
    """Evaluate policy without exploration"""
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        eval_rollout = env.rollout(num_steps, policy)
        eval_reward = eval_rollout["next", "reward"].mean().item()
        return eval_reward

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
    actions_per_episode: int = 480
    episodes_per_batch: int = 20
    desired_num_minibatches: int = 16
    num_epochs: int = 10
    
    # Training parameters
    gamma: float = 0.99
    lr: float = 1e-4
    max_grad_norm: float = 1.0

    def __post_init__(self):
        """Initialize parameter sources and calculate derived values"""
        # Mark all parameters as default initially
        self._param_sources = {
            field_name: 'default' 
            for field_name in self.__dataclass_fields__.keys()
        }
        self._derived_params = {'frames_per_batch', 'total_frames'}
        self._calculate_derived()

    def _calculate_derived(self):
        """Calculate derived parameters"""
        self.frames_per_batch = self.episodes_per_batch * self.actions_per_episode
        self.total_frames = self.episodes * self.actions_per_episode

    @classmethod
    def from_config(cls, cfg: Inifile) -> 'HyperParameters':
        params = cls()
        if 'neuralnetwork-hyperparameters' in cfg.sections():
            section = 'neuralnetwork-hyperparameters'
            for field_name, field in params.__dataclass_fields__.items():
                if field_name not in params._derived_params and cfg.hasopt(section, field_name):
                    if field.type == int:
                        value = cfg.getint(section, field_name)
                    elif field.type == float:
                        value = cfg.getfloat(section, field_name)
                    elif field.type == bool:
                        value = cfg.getbool(section, field_name)
                    else:
                        value = cfg.get(section, field_name)
                    setattr(params, field_name, value)
                    params._param_sources[field_name] = 'config'
        
        params._calculate_derived()
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
                ("actions_per_episode", "Actions per episode"),
                ("episodes_per_batch", "Episodes per update batch"),
                ("desired_num_minibatches", "Target minibatches per update"),
                ("num_epochs", "Training epochs per batch")
            ],
            "Training Parameters": [
                ("gamma", "Discount factor"),
                ("lr", "Learning rate"),
                ("max_grad_norm", "Gradient clipping norm")
            ],
            "Derived Values": [
                ("frames_per_batch", "Frames per batch"),
                ("total_frames", "Total training frames")
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