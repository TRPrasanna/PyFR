from pyfr.inifile import Inifile
from dataclasses import dataclass
from pyfr.readers.native import NativeReader
from typing import Dict, Any
import torch
from torch import nn, optim
from collections import defaultdict
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor, MLP
from torchrl.envs import (
    StepCounter,
    TransformedEnv,
)
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.objectives import CrossQLoss, group_optimizers
from torchrl.modules.models.batchrenorm import BatchRenorm1d
from tqdm.auto import tqdm
from pyfr.rl.env import PyFREnvironment
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import os
import math
from torch.utils.tensorboard import SummaryWriter
import time
import functools
from tensordict import TensorDict
from torchrl.envs.transforms import UnsqueezeTransform, Compose

torch.set_float32_matmul_precision("high")

def train_agent(mesh_file, cfg_file, backend_name, checkpoint_dir='checkpoints', ic_dir=None, load_model=None):
    # Device setup
    #device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
    device = torch.device('cpu')
    #device = torch.device('cuda')

    # Get config path at the start
    if hasattr(cfg_file, 'name'):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    # Initialize environment
    env = PyFREnvironment(mesh_file, cfg_path, backend_name, 0, ic_dir=ic_dir, print_diagnostic=True)
    env = TransformedEnv(
            env,
            Compose(
                UnsqueezeTransform(in_keys=["observation"], dim=0, allow_positive_dim=True), # check, we need this for BatchNorm
                StepCounter(),
            )
        )
    # todo, check: fix PyFR single precision and Pytorch double precision mismatch

    if 'neuralnetwork-hyperparameters' not in env.cfg.sections():
        print("No neuralnetwork-hyperparameters section found in config file. Proceeding to use default hyperparameters.")

    hp = HyperParameters.from_config(env.cfg)
    # Calculate derived parameters using environment info
    hp._calculate_derived(env)
    hp.print_summary()

     # Actor network with proper output handling
    action_spec = env.action_spec_unbatched.to(device)
    actor_hidden_sizes = [hp.num_cells_policy, hp.num_cells_policy]
    actor_net_kwargs = {
        "num_cells": actor_hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": nn.ReLU,
        "norm_class": BatchRenorm1d,
        "norm_kwargs": {
            "momentum": 0.01,
            "num_features": actor_hidden_sizes[-1],
            "warmup_steps": hp.warmup_steps, # not sure, check
        },
    }
    actor_mlp = MLP(**actor_net_kwargs).to(device)

    dist_class = TanhNormal
    dist_kwargs = {
        "low": torch.as_tensor(action_spec.space.low, device=device),
        "high": torch.as_tensor(action_spec.space.high, device=device),
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping="biased_softplus_1.0",
        scale_lb=0.1,
    )
    actor_net = nn.Sequential(actor_mlp, actor_extractor)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=[
            "loc",
            "scale",
        ],
    )
    policy = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    critic_hidden_sizes = [hp.num_cells_value, hp.num_cells_value]
    qvalue_net_kwargs = {
        "num_cells": critic_hidden_sizes,
        "out_features": 1,
        "activation_class": nn.ReLU,
        "norm_class": BatchRenorm1d,
        "norm_kwargs": {
            "momentum": 0.01,
            "num_features": critic_hidden_sizes[-1],
            "warmup_steps": hp.warmup_steps,
        },
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue_module = ValueOperator(
        in_keys=["action"] + ["observation"],
        module=qvalue_net,
    )

    model = nn.ModuleList([policy, qvalue_module]).to(device)
    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict() # check
        td = td.to(device)
        for net in model:
            net.eval()
            net(td)
            net.train()
    del td

    # CrossQ components
    loss_module = CrossQLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function="l2",
        alpha_init=hp.alpha_init,
    )
    loss_module.make_value_estimator(gamma=hp.gamma, device=device)

    # Get number of available devices
    num_devices = get_device_count(backend_name)
    print(f"\nFound {num_devices} devices for backend '{backend_name}'")

    def make_env(backend, device_id):
        """Create environment with specified backend and device ID"""
        base = PyFREnvironment(
            mesh_file=mesh_file,
            cfg_file=cfg_file,
            backend_name=backend,
            device_id=device_id,
            ic_dir=ic_dir,
            print_diagnostic=False
        )
        env = TransformedEnv(
            base,
            Compose(
                UnsqueezeTransform(in_keys=["observation"], dim=0, allow_positive_dim=True), # check, we need this for BatchNorm
                StepCounter(),
            )
        )
        return env

    # Create list of environment creators with device IDs
    env_makers = [
        (lambda id=i: make_env(backend_name, id))
        for i in range(num_devices)
    ]
    collector = MultiSyncDataCollector(
        create_env_fn=env_makers,
        policy=policy,
        frames_per_batch=hp.frames_per_batch,
        total_frames=hp.total_frames,
        init_random_frames=hp.init_random_frames,
        split_trajs=False,
        reset_at_each_iter=True, # without this the collector seems to continue collecting in evaluation mode
        device=device,
    )

    # Replay buffer
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=LazyMemmapStorage(
            hp.replay_buffer_size,
            scratch_dir=None,
        ),
        batch_size=hp.batch_size,
    )
    replay_buffer.append_transform(lambda x: x.to(device, non_blocking=True))

    # optimizers
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=hp.lr,
        weight_decay=0.0,
        eps=1e-8,
        betas=(0.5, 0.999),
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=hp.lr,
        weight_decay=0.0,
        eps=1e-8,
        betas=(0.5, 0.999),
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=hp.alpha_lr,
    )
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    best_eval_reward = float('-inf')
    best_eval_episode = 0
    start_episode = 0
    current_eval_reward = None

    # Load existing model if specified
    if load_model and os.path.exists(load_model):
        checkpoint = torch.load(load_model, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        qvalue_module.load_state_dict(checkpoint['qvalue_state_dict'])
        
        # Load both critic networks from SAC loss module (SAC creates internal copies)
        # Note: The loss_module handles internal critic networks automatically
        # so we don't need to manually load separate critic states
        
        # Load optimizer states if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
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

    # Optional episode progress bar (new)
    try:
        episode_pbar = tqdm(total=hp.episodes, desc="Episodes", leave=False)
        env.set_progress_bar(episode_pbar)
    except:
        print("Warning: Could not create episode progress bar")
        episode_pbar = None
        env.set_progress_bar(None)

    # Tensorboard writer: write different runs based on humean-readable time
    wallclock_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(checkpoint_dir, f"tensorboard_logs/{wallclock_datetime}")
    writer = SummaryWriter(log_dir=log_path)
    # loop through hyperparameters and write them to tensorboard in one go (except _param_sources_ and _derived_params)
    # Collect all hyperparameters first
    hparam_dict = {}
    metric_dict = {}  # Required but empty for hparams logging
    
    for key, value in hp.__dict__.items():
        if key not in ['_param_sources', '_derived_params']:
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
    
    # Write all hyperparameters at once
    run_name = os.path.join(os.path.dirname(os.path.realpath(log_path)),f"{wallclock_datetime}")
    print(f"Writing hyperparameters to tensorboard: {run_name}")
    writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)
    num_updates = int(hp.frames_per_batch * hp.utd_ratio)

    collected_frames = 0
    update_counter = 0
    delayed_updates = hp.policy_update_delay

    for batch_idx, tensordict in enumerate(collector):
        #print(f"\nBatch {i} starting...")
        # Update weights of the inference policy
        collector.update_policy_weights_()

        current_frames = tensordict.numel()

        # Add to replay buffer
        tensordict = tensordict.reshape(-1)
        replay_buffer.extend(tensordict)    

        episode_count += hp.episodes_per_batch
        # Training performance metrics
        train_reward = tensordict["next", "reward"].mean().item()
        writer.add_scalar("batch/train_reward", train_reward, batch_idx)
        writer.add_scalar("batch/episodes", episode_count, batch_idx)
        writer.add_scalar("batch/learning_rate", hp.lr, batch_idx) #change when using lr scheduler

        collected_frames += current_frames
        if collected_frames >= hp.init_random_frames:
            tds = []
            # Optimization steps
            for i in range(num_updates):
                # Update actor every delayed_updates
                update_counter += 1
                update_actor = update_counter % delayed_updates == 0
                sampled_tensordict = replay_buffer.sample().to(device)
                global_update_idx = (batch_idx * num_updates + i) # for logging
                # Compute loss
                if update_actor:
                    optimizer.zero_grad(set_to_none=True)
                    td_loss = {}
                    q_loss, value_meta = loss_module.qvalue_loss(sampled_tensordict)
                    sampled_tensordict.set(loss_module.tensor_keys.priority, value_meta["td_error"])
                    q_loss = q_loss.mean()

                    actor_loss, metadata_actor = loss_module.actor_loss(sampled_tensordict)
                    actor_loss = actor_loss.mean()
                    alpha_loss = loss_module.alpha_loss(
                        log_prob=metadata_actor["log_prob"].detach()
                    ).mean()

                    # Updates
                    (q_loss + actor_loss + alpha_loss).backward()
                    optimizer.step()

                    # Update critic
                    td_loss["loss_qvalue"] = q_loss
                    td_loss["loss_actor"] = actor_loss
                    td_loss["loss_alpha"] = alpha_loss

                    # Log with global update index
                    writer.add_scalar("loss/policy_objective", actor_loss, global_update_idx)
                    writer.add_scalar("loss/alpha_loss", alpha_loss, global_update_idx)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    td_loss = {}
                    q_loss, value_meta = loss_module.qvalue_loss(sampled_tensordict)
                    sampled_tensordict.set(loss_module.tensor_keys.priority, value_meta["td_error"])
                    q_loss = q_loss.mean()

                    # Update critic
                    q_loss.backward()
                    optimizer.step()
                    td_loss["loss_qvalue"] = q_loss
                    td_loss["loss_actor"] = float("nan")
                    td_loss["loss_alpha"] = float("nan")

                tds.append(td_loss.clone())
            
                writer.add_scalar("loss/qvalue_loss", q_loss, global_update_idx)

            tds = TensorDict.stack(tds).nanmean()

        # Logging
        logs["train_reward"].append(train_reward)
        #print(f"\n Batch finished. Episode count is {episode_count}")

        # Evaluate every hp.eval_frequency batches (skip first batch)
        if batch_idx > 0 and batch_idx % hp.eval_frequency == 0:
            eval_reward = evaluate_policy(env, policy)
            logs["eval_reward"].append(eval_reward)

            writer.add_scalar("eval/mean_reward", eval_reward, batch_idx)
            # Possibly log LR
            writer.add_scalar("train/learning_rate", hp.lr, batch_idx)

            # Save best model if new best achieved
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_episode = episode_count
                print(f"\nNew best eval reward: {best_eval_reward:.5f} at episode {episode_count}")
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'qvalue_state_dict': qvalue_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'current_reward': eval_reward,
                    'best_reward': best_eval_reward,
                    'episode': episode_count,
                    'best_episode': best_eval_episode,
                }, best_model_path)

            # Save latest model
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'qvalue_state_dict': qvalue_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
            "lr": f"{hp.lr:.2e}",
        })
        pbar.update(hp.episodes_per_batch)

        #scheduler.step()

    pbar.close()
    if episode_pbar:
        episode_pbar.close()

    collector.shutdown()
    writer.close() # tensorboard writer
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
            #print(f"Eval rewards var: {eval_rollout['next', 'reward']}")
            #print(f"Eval rewards: {eval_reward}")
            del eval_rollout
            return eval_reward
    finally:
        env.set_evaluation_mode(False)  # Reset to training mode
        #print("Evaluation complete.Returning to training mode.")

@dataclass
class HyperParameters:
    # Network architecture
    num_cells_policy: int = 512
    num_cells_value: int = 512
    
    # Training schedule
    episodes: int = 6000
    episodes_per_batch: int = 4
    
    # SAC parameters
    gamma: float = 0.99
    lr: float = 3e-4
    alpha_init: float = 1.0
    alpha_lr: float = 3e-4
    warmup_steps: int = 100000 # Warmup steps for BatchRenorm
    policy_update_delay: int = 3  # Delay for policy updates
    replay_buffer_size: int = 1000000
    init_random_frames: int = 28800 # e.g. frames_per_batch * 24
    batch_size: int = 300 #256
    utd_ratio: float = 1.0  # Update-to-data ratio

    # Evaluation settings
    eval_frequency: int = 5  # Evaluate every N policy updates

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
            ],
            "CrossQ Parameters": [
                ("gamma", "Discount factor"),
                ("lr", "Learning rate for actor/critic"),
                ("alpha_init", "Initial entropy regularization coefficient"),
                ("alpha_lr", "Learning rate for alpha (entropy)"),
                ("warmup_steps", "Warmup steps for BatchRenorm"),
                ("policy_update_delay", "Delay for policy updates"),
                ("replay_buffer_size", "Size of experience replay buffer"),
                ("init_random_frames", "Random frames before training starts"),
                ("batch_size", "Batch size for training"),
                ("utd_ratio", "Update-to-data ratio"),
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

def get_device_count(backend_name):
    """Get number of available devices for given backend"""
    if backend_name == 'hip':
        from pyfr.backends.hip.driver import HIP
        return HIP().device_count()
    elif backend_name == 'cuda':
        from pyfr.backends.cuda.driver import CUDA
        return CUDA().device_count()
    else:
        return 1  # For CPU backends like 'openmp'