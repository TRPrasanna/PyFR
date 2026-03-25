from pyfr.inifile import Inifile
from dataclasses import dataclass
from pyfr.readers.native import NativeReader
from typing import Dict, Any
import torch
from torch import nn
from collections import defaultdict
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor, MLP
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm.auto import tqdm
from pyfr.rl.env import (
    CollectiveEnvController,
    PyFREnvironment,
    serve_collective_envs,
    stop_collective_workers,
)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import os
import math
from torch.utils.tensorboard import SummaryWriter
import time
from pyfr.mpiutil import get_comm_rank_root, init_mpi


class _SingleEnvCollector:
    """Minimal single-process collector to avoid multiprocess GPU duplication."""

    def __init__(self, env, policy, frames_per_batch, total_frames):
        self.env = env
        self.policy = policy
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self._nbatches = math.ceil(total_frames / frames_per_batch)

    def __iter__(self):
        for _ in range(self._nbatches):
            # Match reset_at_each_iter=True behaviour.
            self.env.reset()
            with torch.no_grad():
                batch = self.env.rollout(
                    self.frames_per_batch,
                    self.policy,
                    auto_reset=True,
                    break_when_any_done=False,
                )
            yield batch

    def update_policy_weights_(self):
        return None

    def shutdown(self):
        return None


def train_agent(mesh_file, cfg_file, backend_name,
                checkpoint_dir='checkpoints', ic_dir=None, load_model=None):
    init_mpi()
    comm, rank, root = get_comm_rank_root()
    is_root = rank == root
    collective_mode = comm.size > 1

    # Get config path at the start
    if hasattr(cfg_file, 'name'):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    # Read the config file content, will be later stored in checkpoint
    try:
        with open(cfg_path, 'r') as f:
            config_content = f.read()
    except Exception as e:
        if is_root:
            print(f"Warning: Could not read config file: {e}")
        config_content = None

    # Determine device-id
    device_id = (
        'local-rank'
        if collective_mode and backend_name in {'cuda', 'hip'}
        else 0
    )

    # Initialize environment (MPI-collective: all ranks participate)
    raw_env = PyFREnvironment(
        mesh_file, cfg_path, backend_name, device_id,
        ic_dir=ic_dir, print_diagnostic=is_root
    )

    if is_root:
        if 'neuralnetwork-hyperparameters' not in raw_env.cfg.sections():
            print(
                "No neuralnetwork-hyperparameters section found in config "
                "file. Proceeding to use default hyperparameters."
            )

    hp = HyperParameters.from_config(raw_env.cfg)
    hp._calculate_derived(raw_env)

    # Non-root ranks: serve the collective environment and return
    if collective_mode and not is_root:
        serve_collective_envs({'train': raw_env})
        return

    workers_active = collective_mode

    try:
        # Root only from here
        if collective_mode:
            raw_env = CollectiveEnvController('train', raw_env)

        env = TransformedEnv(raw_env, StepCounter())

        device = torch.device(hp.torch_device)

        # Adjust num_minibatches if it does not divide frames_per_batch evenly
        sub_batch_size = hp.frames_per_batch // hp.desired_num_minibatches
        remainder = hp.frames_per_batch % hp.desired_num_minibatches
        if remainder != 0:
            adjusted_num_minibatches = get_closest_divisor(
                hp.frames_per_batch, hp.desired_num_minibatches
            )
            sub_batch_size = hp.frames_per_batch // adjusted_num_minibatches
            print(
                f"Warning: frames_per_batch ({hp.frames_per_batch}) is not "
                f"perfectly divisible by num_minibatches "
                f"({hp.desired_num_minibatches}). Adjusted "
                f"num_minibatches to {adjusted_num_minibatches} with "
                f"sub_batch_size {sub_batch_size}."
            )
            hp.desired_num_minibatches = adjusted_num_minibatches

        if collective_mode:
            print(
                f'\nMPI collective mode enabled across {comm.size} rank(s). '
                'One RL environment will span the full MPI world.'
            )
            if backend_name in {'cuda', 'hip'}:
                print("Backend device mapping: device-id = 'local-rank'")
        else:
            num_devices = get_device_count(backend_name)
            if num_devices > 1 and backend_name in {'cuda', 'hip'}:
                print(
                    f"\nFound {num_devices} devices for backend "
                    f"'{backend_name}'. To use multiple GPUs, launch one "
                    "MPI rank per GPU via `srun` or `mpiexec`."
                )

        hp.print_summary()

        # Actor network with proper output handling
        action_dim = env.action_spec_unbatched.shape[-1]
        input_shape = env.observation_spec["observation"].shape
        actor_mlp = MLP(
            in_features=input_shape[-1],
            out_features=(
                action_dim if hp.state_ind_normal_scale else 2 * action_dim
            ),
            depth=hp.num_hidden_layers_policy,
            num_cells=hp.num_cells_policy,
            activation_class=getattr(nn, hp.activation_policy),
            device=device,
        )

        # Initialize policy weights
        activation_name = hp.activation_policy
        gain = _safe_gain(activation_name)

        if gain is None:
            print(
                f"Info: Using PyTorch default initialization for actor MLP "
                f"because activation '{activation_name}' has no supported gain."
            )
        else:
            for layer in actor_mlp.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

        # Add learnable scales (standard deviations)
        if hp.state_ind_normal_scale:
            actor_net = nn.Sequential(
                actor_mlp,
                AddStateIndependentNormalScale(
                    action_dim,
                    scale_lb=1e-8,
                ).to(device)
            )
        else:
            actor_net = nn.Sequential(
                actor_mlp,
                NormalParamExtractor(
                    scale_mapping="biased_softplus_1.0",
                    scale_lb=0.1,
                ).to(device)
            )

        actor_module = TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"]
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
                "tanh_loc": False,
            },
        ).to(device)

        # Value network (critic)
        value_net = MLP(
            in_features=input_shape[-1],
            out_features=1,
            depth=hp.num_hidden_layers_value,
            num_cells=hp.num_cells_value,
            activation_class=getattr(nn, hp.activation_value),
            device=device,
        )

        # Initialize value weights
        activation_name = hp.activation_value
        gain = _safe_gain(activation_name)

        if gain is None:
            print(
                f"Info: Using PyTorch default initialization for value MLP "
                f"because activation '{activation_name}' has no supported gain."
            )
        else:
            for layer in value_net.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

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
            entropy_coeff=hp.entropy_eps,
            critic_coeff=1.0,
            loss_critic_type="smooth_l1",
        )

        # Optimizer
        optim = torch.optim.Adam(loss_module.parameters(), hp.lr)

        print(
            f"\nUsing single-environment collector "
            f"({'MPI collective' if collective_mode else 'local'})."
        )

        collector = _SingleEnvCollector(
            env=env,
            policy=policy,
            frames_per_batch=hp.frames_per_batch,
            total_frames=hp.total_frames,
        )

        # Replay buffer (used for minibatch sampling, not experience replay)
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=hp.frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        best_eval_reward = float('-inf')
        best_eval_episode = 0
        start_episode = 0
        current_eval_reward = None

        start_batch_idx = 0
        # Load existing model if specified
        if load_model and os.path.exists(load_model):
            checkpoint = torch.load(
                load_model, map_location=device, weights_only=True
            )
            policy.load_state_dict(checkpoint['policy_state_dict'])
            value_module.load_state_dict(checkpoint['value_state_dict'])

            current_eval_reward = checkpoint.get(
                'current_reward', float('-inf')
            )
            loaded_best_reward = checkpoint.get(
                'best_reward', float('-inf')
            )
            start_episode = checkpoint.get('episode', 0)
            loaded_best_episode = checkpoint.get('best_episode', 0)
            start_batch_idx = checkpoint.get('batch_idx', 0) + 1

            # Get saved hyperparameters and compare with current
            saved_hp = checkpoint.get('hyperparameters', {})
            differences = []

            if saved_hp:
                print("\nVerifying hyperparameters...")
                for key, saved_value in saved_hp.items():
                    if hasattr(hp, key):
                        current_value = getattr(hp, key)
                        if current_value != saved_value:
                            differences.append(
                                (key, saved_value, current_value)
                            )

            if differences:
                key_width = 22
                val_width = 20

                key_sep = '─' * (key_width + 2)
                val_sep = '─' * (val_width + 2)

                print(
                    "\nWARNING: Hyperparameter differences detected "
                    "between checkpoint and current settings:"
                )
                print(f"┌{key_sep}┬{val_sep}┬{val_sep}┐")
                print(
                    f"│ {'Key':<{key_width}} │ "
                    f"{'Checkpoint Value':<{val_width}} │ "
                    f"{'Current Value':<{val_width}} │"
                )
                print(f"├{key_sep}┼{val_sep}┼{val_sep}┤")
                for key, saved, current in differences:
                    print(
                        f"│ {key:<{key_width}} │ "
                        f"{str(saved):<{val_width}} │ "
                        f"{str(current):<{val_width}} │"
                    )
                print(f"└{key_sep}┴{val_sep}┴{val_sep}┘")
            else:
                print("done.")

            # Compare config files if available
            if 'config_content' in checkpoint and config_content:
                print("\nVerifying config files...")
                config_differences = compare_configs(
                    checkpoint['config_content'], config_content
                )

                if config_differences:
                    print(
                        "\nWARNING: Config file differences detected "
                        "between checkpoint and current:"
                    )
                    for line_num, ckpt_line, curr_line in config_differences:
                        print(f"Line {line_num}:")
                        print(f"  Checkpoint: {ckpt_line}")
                        print(f"  Current:    {curr_line}")
                        print()
                else:
                    print(
                        "Config files match between checkpoint "
                        "and current settings."
                    )

            if (hasattr(hp, 'print_config_on_load')
                    and hp.print_config_on_load
                    and 'config_content' in checkpoint):
                print("\n=== CHECKPOINT CONFIG FILE CONTENT ===\n")
                print(checkpoint['config_content'])
                print("\n=======================================\n")

            print(f"\nLoaded model from: {load_model}")
            print(f"Current eval reward: {current_eval_reward:.4f}")
            print(
                f"Best eval reward from checkpoint: "
                f"{loaded_best_reward:.4f}"
            )
            print(
                f"Best reward achieved at episode: {loaded_best_episode}"
            )
            print(f"Continuing from episode: {start_episode}\n")

            if (isinstance(loaded_best_reward, (int, float))
                    and loaded_best_reward > best_eval_reward):
                best_eval_reward = loaded_best_reward
                best_eval_episode = loaded_best_episode
                print(
                    f"Updated best reward tracking to: "
                    f"{best_eval_reward:.4f}\n"
                )

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, 'best-model.pt')
        latest_model_path = os.path.join(checkpoint_dir, 'latest-model.pt')
        logs = defaultdict(list)
        remaining_episodes = hp.episodes - start_episode
        pbar = tqdm(
            total=remaining_episodes, desc="Training",
            initial=start_episode
        )
        episode_count = start_episode

        eval_str = ""

        # Tensorboard writer
        wallclock_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(
            checkpoint_dir, f"tensorboard_logs/{wallclock_datetime}"
        )
        writer = SummaryWriter(log_dir=log_path)

        hparam_dict = {}
        for key, value in hp.__dict__.items():
            if key not in ['_param_sources', '_derived_params']:
                if isinstance(value, (int, float, str, bool)):
                    hparam_dict[key] = value

        run_name = os.path.join(
            os.path.dirname(os.path.realpath(log_path)),
            f"{wallclock_datetime}"
        )
        print(
            f"Writing hyperparameters to tensorboard: {run_name}"
        )
        writer.add_hparams(hparam_dict, {}, run_name=run_name)

        updates_per_batch = (
            hp.num_epochs * (hp.frames_per_batch // sub_batch_size)
        )

        batch_idx = start_batch_idx
        for _, tensordict_data in enumerate(collector):
            episode_count += hp.episodes_per_batch

            # Training performance metrics
            train_reward = (
                tensordict_data["next", "reward"].mean().item()
            )
            writer.add_scalar(
                "batch/train_reward", train_reward, batch_idx
            )
            writer.add_scalar(
                "batch/episodes", episode_count, batch_idx
            )
            writer.add_scalar(
                "batch/learning_rate",
                optim.param_groups[0]['lr'], batch_idx
            )

            # Training updates
            for epoch_idx in range(hp.num_epochs):
                advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view.cpu())

                for sub_update_idx in range(
                    hp.frames_per_batch // sub_batch_size
                ):
                    subdata = replay_buffer.sample(sub_batch_size)
                    loss_vals = loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                    )
                    if hp.entropy_eps > 0:
                        loss_value = loss_value + loss_vals["loss_entropy"]

                    policy_obj = loss_vals["loss_objective"].item()
                    val_loss = loss_vals["loss_critic"].item()
                    ent_loss = (
                        loss_vals.get("loss_entropy", 0.0).item()
                        if isinstance(
                            loss_vals.get("loss_entropy", 0.0),
                            torch.Tensor
                        )
                        else 0.0
                    )

                    loss_value.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        loss_module.parameters(), hp.max_grad_norm
                    )

                    global_update_idx = (
                        batch_idx * updates_per_batch
                        + epoch_idx
                        * (hp.frames_per_batch // sub_batch_size)
                        + sub_update_idx
                    )

                    writer.add_scalar(
                        "loss/policy_objective",
                        policy_obj, global_update_idx
                    )
                    writer.add_scalar(
                        "loss/value_loss",
                        val_loss, global_update_idx
                    )
                    writer.add_scalar(
                        "loss/entropy_bonus",
                        ent_loss, global_update_idx
                    )
                    writer.add_scalar(
                        "grad/norm",
                        grad_norm, global_update_idx
                    )

                    optim.step()
                    optim.zero_grad()

            collector.update_policy_weights_()

            # Logging
            logs["train_reward"].append(train_reward)

            # Evaluate every hp.eval_frequency batches
            if batch_idx % hp.eval_frequency == 0:
                eval_reward = evaluate_policy(env, policy)
                logs["eval_reward"].append(eval_reward)

                writer.add_scalar(
                    "eval/mean_reward", eval_reward, batch_idx + 1
                )
                writer.add_scalar(
                    "train/learning_rate",
                    optim.param_groups[0]['lr'], batch_idx + 1
                )

                # Save checkpoint dict helper
                def _ckpt_dict(reward):
                    return {
                        'policy_state_dict': policy.state_dict(),
                        'value_state_dict': value_module.state_dict(),
                        'current_reward': reward,
                        'best_reward': best_eval_reward,
                        'episode': episode_count,
                        'best_episode': best_eval_episode,
                        'batch_idx': batch_idx,
                        'hyperparameters': {
                            k: v for k, v in hp.__dict__.items()
                            if not k.startswith('_') and not callable(v)
                        },
                        'config_content': config_content,
                        'config_path': cfg_path,
                    }

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_eval_episode = episode_count
                    print(
                        f"\nNew best eval reward: "
                        f"{best_eval_reward:.5f} at episode "
                        f"{episode_count}"
                    )
                    torch.save(_ckpt_dict(eval_reward), best_model_path)

                torch.save(
                    _ckpt_dict(eval_reward),
                    os.path.join(
                        checkpoint_dir, f'model-{batch_idx + 1}.pt'
                    )
                )
                torch.save(_ckpt_dict(eval_reward), latest_model_path)

                eval_str = (
                    f"eval reward: {eval_reward:.5f} "
                    f"(best: {best_eval_reward:.5f})"
                )

            # Progress bar update
            pbar.set_postfix({
                "train_reward": f"{train_reward:.5f}",
                "eval": eval_str,
                "lr": f"{optim.param_groups[0]['lr']:.2e}",
            })
            pbar.update(hp.episodes_per_batch)

            batch_idx += 1

        pbar.close()
        collector.shutdown()
        writer.close()
    finally:
        try:
            env.close()
        except Exception:
            try:
                raw_env.close(raise_if_closed=False)
            except Exception:
                pass

        if workers_active:
            stop_collective_workers(comm, root)


def evaluate_policy(env, policy, num_steps=1000000):
    """Evaluate policy without exploration using consistent IC"""
    env.set_evaluation_mode(True)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), \
                torch.no_grad():
            eval_rollout = env.rollout(num_steps, policy)
            eval_reward = eval_rollout["next", "reward"].mean().item()
            del eval_rollout
            return eval_reward
    finally:
        env.set_evaluation_mode(False)


def _safe_gain(act_name: str):
    name = (act_name or "").lower()
    if name in {"leakyrelu", "leaky_relu"}:
        try:
            return torch.nn.init.calculate_gain("leaky_relu", 0.01)
        except Exception:
            return None
    valid = {
        "linear", "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "sigmoid", "tanh", "relu", "leaky_relu", "selu"
    }
    if name in valid:
        try:
            return torch.nn.init.calculate_gain(name)
        except Exception:
            return None
    return None


def get_closest_divisor(n, target):
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)

    closest = min(divisors, key=lambda x: (abs(x - target), -x))
    return closest


def compare_configs(checkpoint_config, current_config):
    if not checkpoint_config or not current_config:
        return []

    checkpoint_lines = [
        line.strip() for line in checkpoint_config.splitlines()
    ]
    current_lines = [
        line.strip() for line in current_config.splitlines()
    ]

    differences = []

    for i, (ckpt_line, curr_line) in enumerate(
        zip(checkpoint_lines, current_lines)
    ):
        if (not ckpt_line or ckpt_line.startswith(';')
                or not curr_line or curr_line.startswith(';')):
            continue

        if ckpt_line != curr_line:
            differences.append((i + 1, ckpt_line, curr_line))

    if len(checkpoint_lines) > len(current_lines):
        for i, line in enumerate(
            checkpoint_lines[len(current_lines):],
            start=len(current_lines)
        ):
            if line and not line.startswith(';'):
                differences.append((i + 1, line, "[MISSING]"))

    elif len(current_lines) > len(checkpoint_lines):
        for i, line in enumerate(
            current_lines[len(checkpoint_lines):],
            start=len(checkpoint_lines)
        ):
            if line and not line.startswith(';'):
                differences.append((i + 1, "[MISSING]", line))

    return differences


@dataclass
class HyperParameters:
    # General settings
    torch_device: str = 'cpu'
    print_config_on_load: bool = False
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
    eval_frequency: int = 1

    def __post_init__(self):
        self._param_sources = {
            field_name: 'default'
            for field_name in self.__dataclass_fields__.keys()
        }
        self._derived_params = {
            'frames_per_batch', 'total_frames', 'actions_per_episode'
        }
        self.actions_per_episode = None
        self.frames_per_batch = None
        self.total_frames = None

    def _calculate_derived(self, env):
        self.actions_per_episode = int(env.dtend / env.action_interval)
        self.frames_per_batch = (
            self.episodes_per_batch * self.actions_per_episode
        )
        self.total_frames = self.episodes * self.actions_per_episode

    @classmethod
    def from_config(cls, cfg: Inifile) -> 'HyperParameters':
        params = cls()
        if 'neuralnetwork-hyperparameters' in cfg.sections():
            section = 'neuralnetwork-hyperparameters'
            for field_name, field in params.__dataclass_fields__.items():
                config_name = field_name.replace('_', '-')
                if (field_name not in params._derived_params
                        and cfg.hasopt(section, config_name)):
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
        sections = {
            "General Settings": [
                ("torch_device", "'cuda' or 'cpu'"),
                ("print_config_on_load",
                 "Print config file content on model load"),
            ],
            "Network Architecture": [
                ("num_hidden_layers_policy",
                 "No. of hidden layers in policy network"),
                ("num_hidden_layers_value",
                 "No. of hidden layers in value network"),
                ("num_cells_policy",
                 "Size of policy network hidden layers"),
                ("num_cells_value",
                 "Size of value network hidden layers"),
                ("activation_policy",
                 "Activation function for policy network"),
                ("activation_value",
                 "Activation function for value network"),
                ("state_ind_normal_scale",
                 "state-independent normal scale for actions"),
            ],
            "Training Schedule": [
                ("episodes", "Total training episodes"),
                ("episodes_per_batch", "Episodes per update batch"),
                ("desired_num_minibatches",
                 "Target minibatches per update"),
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

        param_width = 25
        value_width = 15
        src_width = 5
        desc_width = 40

        def wrap_text(text, width):
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

        top_line = (
            f"{tl_corner}{h_line * (param_width + 2)}"
            f"{t_down}{h_line * (value_width + 2)}"
            f"{t_down}{h_line * (src_width + 2)}"
            f"{t_down}{h_line * (desc_width + 2)}{tr_corner}"
        )
        mid_line = (
            f"{t_right}{h_line * (param_width + 2)}"
            f"{cross}{h_line * (value_width + 2)}"
            f"{cross}{h_line * (src_width + 2)}"
            f"{cross}{h_line * (desc_width + 2)}{t_left}"
        )
        bot_line = (
            f"{bl_corner}{h_line * (param_width + 2)}"
            f"{t_up}{h_line * (value_width + 2)}"
            f"{t_up}{h_line * (src_width + 2)}"
            f"{t_up}{h_line * (desc_width + 2)}{br_corner}"
        )

        def format_row(param, value, source, desc_line,
                        is_continuation=False):
            src_display = "" if is_continuation else source
            return (
                f"{v_line} {param:<{param_width}} "
                f"{v_line} {str(value):<{value_width}} "
                f"{v_line} {src_display:<{src_width}} "
                f"{v_line} {desc_line:<{desc_width}} {v_line}"
            )

        def format_header():
            return (
                f"{v_line} {'Parameter':<{param_width}} "
                f"{v_line} {'Value':<{value_width}} "
                f"{v_line} {'Src':<{src_width}} "
                f"{v_line} {'Description':<{desc_width}} {v_line}"
            )

        print("\nHyperparameters Configuration")

        for section_name, params in sections.items():
            print(f"\n{section_name}:")
            print(top_line)
            print(format_header())
            print(mid_line)

            for param_name, description in params:
                value = getattr(self, param_name)

                if param_name in self._derived_params:
                    source = "[-]"
                elif (param_name in self._param_sources
                        and self._param_sources[param_name] == "config"):
                    source = "[C]"
                else:
                    source = "[D]"

                desc_lines = wrap_text(description, desc_width)

                print(format_row(
                    param_name, value, source, desc_lines[0]
                ))

                for line in desc_lines[1:]:
                    print(format_row(
                        "", "", "", line, is_continuation=True
                    ))

            print(bot_line)

        print(
            "\nSource: [C]=From .ini config file, "
            "[D]=Default, [-]=Derived"
        )


def get_device_count(backend_name):
    if backend_name == 'hip':
        from pyfr.backends.hip.driver import HIP
        return HIP().device_count()
    elif backend_name == 'cuda':
        from pyfr.backends.cuda.driver import CUDA
        return CUDA().device_count()
    else:
        return 1
