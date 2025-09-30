# train.py  (PPO-LSTM)
from dataclasses import dataclass
from typing import Any
import os, time
from collections import defaultdict

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    NormalParamExtractor,
    LSTMModule,
    MLP,
    set_recurrent_mode,  # correct import location
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs import TransformedEnv, Compose, StepCounter, InitTracker
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.collectors import MultiSyncDataCollector
from tqdm.auto import tqdm

from pyfr.inifile import Inifile
from pyfr.rl.env import PyFREnvironment

# ---------------------------
# Utilities and hyperparams
# ---------------------------

def get_device_count(backend_name):
    if backend_name == 'hip':
        from pyfr.backends.hip.driver import HIP
        return HIP().device_count()
    elif backend_name == 'cuda':
        from pyfr.backends.cuda.driver import CUDA
        return CUDA().device_count()
    else:
        return 1

@dataclass
class HyperParameters:
    # General
    torch_device: str = "cpu"
    print_config_on_load: bool = False

    # Architecture
    num_cells_policy: int = 512   # LSTM hidden size (actor)
    num_cells_value: int  = 512   # LSTM hidden size (critic)
    activation_policy: str = "Tanh"
    activation_value: str  = "Tanh"

    # Schedule
    episodes: int = 5000
    episodes_per_batch: int = 20
    num_epochs: int = 10

    # PPO
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.97
    entropy_eps: float = 1e-3
    lr: float = 1e-4
    max_grad_norm: float = 10.0

    # Eval
    eval_frequency: int = 1

    # Derived
    actions_per_episode: int = None
    frames_per_batch: int = None
    total_frames: int = None

    # Optional config loader hook (retain if you have one)
    @classmethod
    def from_config(cls, cfg: Inifile) -> "HyperParameters":
        params = cls()
        if 'neuralnetwork-hyperparameters' in cfg.sections():
            section = 'neuralnetwork-hyperparameters'
            for field_name, field in params.__dataclass_fields__.items():
                cfg_key = field_name.replace('_', '-')
                if cfg.hasopt(section, cfg_key):
                    if field.type == int:
                        val = cfg.getint(section, cfg_key)
                    elif field.type == float:
                        val = cfg.getfloat(section, cfg_key)
                    elif field.type == bool:
                        val = cfg.getbool(section, cfg_key)
                    else:
                        val = cfg.get(section, cfg_key)
                    setattr(params, field_name, val)
        return params

    def _calculate_derived(self, env):
        self.actions_per_episode = int(env.dtend / env.action_interval)
        self.frames_per_batch = self.episodes_per_batch * self.actions_per_episode
        self.total_frames = self.episodes * self.actions_per_episode

# ---------------------------
# Evaluation
# ---------------------------

def evaluate_policy(env, policy, num_steps=1_000_000):
    env.set_evaluation_mode(True)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            rollout = env.rollout(num_steps, policy)
            return rollout["next", "reward"].mean().item()
    finally:
        env.set_evaluation_mode(False)

# ---------------------------
# Main training
# ---------------------------

def train_agent(mesh_file, cfg_file, backend_name, checkpoint_dir='checkpoints', ic_dir=None, load_model=None):
    # Resolve config path and cache text for checkpoint diff
    cfg_path = cfg_file.name if hasattr(cfg_file, 'name') else cfg_file
    try:
        with open(cfg_path, 'r') as f:
            config_content = f.read()
    except Exception:
        config_content = None

    # Base env with StepCounter + InitTracker
    base_env = PyFREnvironment(mesh_file, cfg_path, backend_name, 0, ic_dir=ic_dir, print_diagnostic=True)
    env = TransformedEnv(base_env, Compose(StepCounter(), InitTracker()))

    # Hyperparameters
    if 'neuralnetwork-hyperparameters' not in env.cfg.sections():
        print("No [neuralnetwork-hyperparameters] in cfg. Using defaults.")
    hp = HyperParameters.from_config(env.cfg)
    hp._calculate_derived(env)
    device = torch.device(hp.torch_device)

    # Shapes
    action_dim = env.action_spec_unbatched.shape[-1]
    obs_dim    = env.observation_spec["observation"].shape[-1]

    # ---------------------------
    # Actor: LSTM + head -> loc, scale
    # ---------------------------
    actor_lstm = LSTMModule(
        input_size=obs_dim,
        hidden_size=hp.num_cells_policy,
        device=device,
        in_keys=["observation", "actor_h", "actor_c"],
        out_keys=["actor_feat", ("next", "actor_h"), ("next", "actor_c")],
        python_based=True,
    )
    actor_head = MLP(
        in_features=hp.num_cells_policy,
        out_features=2 * action_dim,
        num_cells=[hp.num_cells_policy],
        activation_class=getattr(nn, hp.activation_policy),
        device=device,
    )
    actor_head_mod = TensorDictModule(actor_head, in_keys=["actor_feat"], out_keys=["actor_params"])
    param_extract  = TensorDictModule(
        NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=0.1),
        in_keys=["actor_params"], out_keys=["loc", "scale"]
    )
    actor_net = TensorDictSequential(actor_lstm, actor_head_mod, param_extract).to(device)
    policy = ProbabilisticActor(
        module=actor_net,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        distribution_kwargs={
            "low":  env.action_spec.space.low,
            "high": env.action_spec.space.high,
            "tanh_loc": False,
        },
    ).to(device)

    # ---------------------------
    # Critic: LSTM + head -> state_value
    # ---------------------------
    critic_lstm = LSTMModule(
        input_size=obs_dim,
        hidden_size=hp.num_cells_value,
        device=device,
        in_keys=["observation", "critic_h", "critic_c"],
        out_keys=["critic_feat", ("next", "critic_h"), ("next", "critic_c")],
        python_based=True,
    )
    critic_head = MLP(
        in_features=hp.num_cells_value,
        out_features=1,
        num_cells=[hp.num_cells_value],
        activation_class=getattr(nn, hp.activation_value),
        device=device,
    )
    critic_head_mod = TensorDictModule(critic_head, in_keys=["critic_feat"], out_keys=["state_value"])
    value_net = TensorDictSequential(critic_lstm, critic_head_mod).to(device)

    # Register both primers
    env.append_transform(actor_lstm.make_tensordict_primer())
    env.append_transform(critic_lstm.make_tensordict_primer())

    # ---------------------------
    # Advantage + PPO loss
    # ---------------------------
    advantage_module = GAE(
        gamma=hp.gamma,
        lmbda=hp.lmbda,
        value_network=value_net,
        average_gae=True,
        deactivate_vmap=True,
    )
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_net,
        clip_epsilon=hp.clip_epsilon,
        entropy_bonus=bool(hp.entropy_eps),
        entropy_coeff=hp.entropy_eps,   # updated name
        critic_coeff=1.0,               # updated name
        loss_critic_type="smooth_l1",
    )
    optim = torch.optim.Adam(loss_module.parameters(), hp.lr)

    # ---------------------------
    # Collector
    # ---------------------------
    num_devices = get_device_count(backend_name)
    print(f"Found {num_devices} devices for backend '{backend_name}'")

    def make_env(backend, device_id):
        e = PyFREnvironment(mesh_file, cfg_file, backend, device_id, ic_dir=ic_dir, print_diagnostic=False)
        e = TransformedEnv(e, Compose(StepCounter(), InitTracker()))
        e.append_transform(actor_lstm.make_tensordict_primer())
        e.append_transform(critic_lstm.make_tensordict_primer())
        return e

    env_makers = [(lambda j=i: make_env(backend_name, j)) for i in range(num_devices)]

    collector = MultiSyncDataCollector(
        create_env_fn=env_makers,
        policy=policy,
        frames_per_batch=hp.frames_per_batch,
        total_frames=hp.total_frames,
        reset_at_each_iter=True,
        device=device,
    )

    # ---------------------------
    # Checkpointing and logging
    # ---------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path   = os.path.join(checkpoint_dir, "best_model.pt")
    latest_model_path = os.path.join(checkpoint_dir, "latest_model.pt")

    best_eval_reward = float("-inf")
    best_eval_episode = 0
    episode_count = 0

    if load_model and os.path.exists(load_model):
        ckpt = torch.load(load_model, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy_state_dict"])
        value_net.load_state_dict(ckpt["value_state_dict"])
        print(f"Loaded model: {load_model}")
        if "best_reward" in ckpt:
            best_eval_reward = ckpt["best_reward"]
            best_eval_episode = ckpt.get("best_episode", 0)

    wallclock = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(checkpoint_dir, f"tensorboard_logs/{wallclock}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Writing hyperparameters to tensorboard: {log_dir}")
    hparam_dict = {k: v for k, v in hp.__dict__.items() if isinstance(v, (int, float, str, bool))}
    writer.add_hparams(hparam_dict, {}, run_name=log_dir)

    # ---------------------------
    # Train loop
    # ---------------------------
    pbar = tqdm(total=hp.episodes, desc="Training")
    updates_per_batch = hp.num_epochs

    for batch_idx, td in enumerate(collector):
        episode_count += hp.episodes_per_batch

        train_reward = td["next", "reward"].mean().item()
        writer.add_scalar("batch/train_reward", train_reward, batch_idx)
        writer.add_scalar("batch/episodes", episode_count, batch_idx)
        writer.add_scalar("batch/learning_rate", optim.param_groups[0]["lr"], batch_idx)

        for ep in range(hp.num_epochs):
            with set_recurrent_mode(True):
                advantage_module(td)
            with set_recurrent_mode(True):
                loss_vals = loss_module(td.to(device))
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"]
                if hp.entropy_eps > 0:
                    loss_value = loss_value + loss_vals["loss_entropy"]

            loss_value.backward()
            grad_norm = nn.utils.clip_grad_norm_(loss_module.parameters(), hp.max_grad_norm)
            optim.step()
            optim.zero_grad()

            global_update = batch_idx * updates_per_batch + ep
            writer.add_scalar("loss/policy_objective", loss_vals["loss_objective"].item(), global_update)
            writer.add_scalar("loss/value_loss",      loss_vals["loss_critic"].item(),    global_update)
            writer.add_scalar(
                "loss/entropy_bonus",
                loss_vals.get("loss_entropy", torch.tensor(0.0)).detach().item(),
                global_update,
            )
            writer.add_scalar("grad/norm", float(grad_norm if not isinstance(grad_norm, torch.Tensor) else grad_norm.detach().item()), global_update)


        collector.update_policy_weights_()

        if batch_idx % hp.eval_frequency == 0:
            eval_reward = evaluate_policy(env, policy)
            writer.add_scalar("eval/mean_reward", eval_reward, batch_idx)
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_episode = episode_count
                print(f"New best eval reward: {best_eval_reward:.5f} at episode {episode_count}")
                torch.save({
                    "policy_state_dict": policy.state_dict(),
                    "value_state_dict": value_net.state_dict(),
                    "current_reward": eval_reward,
                    "best_reward": best_eval_reward,
                    "best_episode": best_eval_episode,
                    "config_content": config_content,
                    "config_path": cfg_path,
                }, best_model_path)
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "value_state_dict": value_net.state_dict(),
                "current_reward": eval_reward,
                "best_reward": best_eval_reward,
                "best_episode": best_eval_episode,
                "config_content": config_content,
                "config_path": cfg_path,
            }, latest_model_path)

        pbar.set_postfix({
            "train_reward": f"{train_reward:.5f}",
            "lr": f"{optim.param_groups[0]['lr']:.2e}",
        })
        pbar.update(hp.episodes_per_batch)

    pbar.close()
    collector.shutdown()
    writer.close()
    env.close()

def compare_configs(checkpoint_config, current_config):
    """
    Compare two config files line by line and return differences.
    
    Args:
        checkpoint_config: Config content from checkpoint as string
        current_config: Current config content as string
        
    Returns:
        List of tuples with (line_number, checkpoint_line, current_line) for different lines
    """
    if not checkpoint_config or not current_config:
        return []
        
    # Split into lines and strip whitespace
    checkpoint_lines = [line.strip() for line in checkpoint_config.splitlines()]
    current_lines = [line.strip() for line in current_config.splitlines()]
    
    # Find differences
    differences = []
    
    # First, check lines that exist in both files
    for i, (ckpt_line, curr_line) in enumerate(zip(checkpoint_lines, current_lines)):
        # Skip empty lines and comments
        if not ckpt_line or ckpt_line.startswith(';') or not curr_line or curr_line.startswith(';'):
            continue
            
        if ckpt_line != curr_line:
            differences.append((i+1, ckpt_line, curr_line))
    
    # Check if one file has more lines than the other
    if len(checkpoint_lines) > len(current_lines):
        for i, line in enumerate(checkpoint_lines[len(current_lines):], start=len(current_lines)):
            if line and not line.startswith(';'):  # Skip empty lines and comments
                differences.append((i+1, line, "[MISSING]"))
    
    elif len(current_lines) > len(checkpoint_lines):
        for i, line in enumerate(current_lines[len(checkpoint_lines):], start=len(checkpoint_lines)):
            if line and not line.startswith(';'):  # Skip empty lines and comments
                differences.append((i+1, "[MISSING]", line))
    
    return differences
@dataclass
class HyperParameters:
    # General settings
    torch_device: str = 'cpu'  # 'cuda', 'cpu'
    print_config_on_load: bool = False # set to True to view config file content on model load
    # Network architecture
    num_hidden_layers_policy: int = 2
    num_hidden_layers_value: int = 2
    num_cells_policy: int = 512
    num_cells_value: int = 512
    activation_policy: str = 'Tanh'
    activation_value: str = 'Tanh'
    state_ind_normal_scale: bool = False
    
    # Training schedule
    episodes: int = 1200
    episodes_per_batch: int = 20
    desired_num_minibatches: int = 20
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
            "General Settings": [
                ("torch_device", "'cuda' or 'cpu'"),
                ("print_config_on_load", "Print config file content on model load"),
            ],
            "Network Architecture": [
                ("num_hidden_layers_policy", "No. of hidden layers in policy network"),
                ("num_hidden_layers_value", "No. of hidden layers in value network"),
                ("num_cells_policy", "Size of policy network hidden layers"),
                ("num_cells_value", "Size of value network hidden layers"),
                ("activation_policy", "Activation function for policy network"),
                ("activation_value", "Activation function for value network"),
                ("state_ind_normal_scale", "state-independent normal scale for actions"),
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

        # Define consistent column widths
        param_width = 25
        value_width = 15
        src_width = 5
        desc_width = 40
        
        # Line breaking function for descriptions
        def wrap_text(text, width):
            """Wrap text to fit within width"""
            if len(text) <= width:
                return [text]
            
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + len(current_line) <= width:
                    current_line.append(word)
                    current_length += len(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            return lines
        
        # Box drawing characters for continuous tables
        h_line = "─"
        v_line = "│"
        tl_corner = "┌"
        tr_corner = "┐"
        bl_corner = "└"
        br_corner = "┘"
        t_down = "┬"
        t_up = "┴"
        t_right = "├"
        t_left = "┤"
        cross = "┼"
        
        # Create horizontal lines
        top_line = f"{tl_corner}{h_line * (param_width + 2)}{t_down}{h_line * (value_width + 2)}{t_down}{h_line * (src_width + 2)}{t_down}{h_line * (desc_width + 2)}{tr_corner}"
        mid_line = f"{t_right}{h_line * (param_width + 2)}{cross}{h_line * (value_width + 2)}{cross}{h_line * (src_width + 2)}{cross}{h_line * (desc_width + 2)}{t_left}"
        bot_line = f"{bl_corner}{h_line * (param_width + 2)}{t_up}{h_line * (value_width + 2)}{t_up}{h_line * (src_width + 2)}{t_up}{h_line * (desc_width + 2)}{br_corner}"
        
        def format_row(param, value, source, desc_line, is_continuation=False):
            """Format a single row of the table"""
            # Leave source empty for continuation lines
            src_display = "" if is_continuation else source
            return f"{v_line} {param:<{param_width}} {v_line} {str(value):<{value_width}} {v_line} {src_display:<{src_width}} {v_line} {desc_line:<{desc_width}} {v_line}"

        def format_header():
            """Format the table header with correct column names"""
            return f"{v_line} {'Parameter':<{param_width}} {v_line} {'Value':<{value_width}} {v_line} {'Src':<{src_width}} {v_line} {'Description':<{desc_width}} {v_line}"

        print("\nHyperparameters Configuration")
        
        for section_name, params in sections.items():
            print(f"\n{section_name}:")
            print(top_line)
            print(format_header())
            print(mid_line)
            
            for param_name, description in params:
                value = getattr(self, param_name)
                
                # Determine source marker
                if param_name in self._derived_params:
                    source = "[-]"
                elif param_name in self._param_sources and self._param_sources[param_name] == "config":
                    source = "[C]"
                else:
                    source = "[D]"
                
                # Handle multi-line descriptions
                desc_lines = wrap_text(description, desc_width)
                
                # Print first line with parameter info
                print(format_row(param_name, value, source, desc_lines[0]))
                
                # Print continuation lines if any
                for line in desc_lines[1:]:
                    # Empty strings for param and value, and is_continuation=True to not show source
                    print(format_row("", "", "", line, is_continuation=True))
                
            print(bot_line)

        # Print legend
        print("\nSource: [C]=From .ini config file, [D]=Default, [-]=Derived")