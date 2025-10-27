from dataclasses import dataclass

import math
import os
import time
from collections import defaultdict

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tensordict.nn import (
    AddStateIndependentNormalScale,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl.collectors import MultiSyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import Compose, InitTracker, StepCounter, TransformedEnv
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    LSTMModule,
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules.utils import get_primers_from_module
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm.auto import tqdm

from pyfr.inifile import Inifile
from pyfr.rl.env import PyFREnvironment


def train_agent(
    mesh_file,
    cfg_file,
    backend_name,
    checkpoint_dir="checkpoints",
    ic_dir=None,
    load_model=None,
):
    # Get config path at the start
    if hasattr(cfg_file, "name"):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    # Read config file content now (for checkpoint reproducibility)
    try:
        with open(cfg_path, "r") as f:
            config_content = f.read()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not read config file: {e}")
        config_content = None

    # =========================
    # Env init (root env first)
    # =========================
    env = PyFREnvironment(
        mesh_file, cfg_path, backend_name, 0, ic_dir=ic_dir, print_diagnostic=True
    )
    # Wrap with:
    # - StepCounter() gives per-episode step idx
    # - InitTracker() marks reset boundaries ("is_init"), needed for recurrent state resets
    env = TransformedEnv(env, Compose(StepCounter(), InitTracker()))

    if "neuralnetwork-hyperparameters" not in env.cfg.sections():
        print(
            "No neuralnetwork-hyperparameters section found in config file. "
            "Proceeding to use default hyperparameters."
        )

    hp = HyperParameters.from_config(env.cfg)
    hp._calculate_derived(env)

    device = torch.device(hp.torch_device)

    hp.print_summary()

    # ==================================
    # Actor-Critic recurrent architectures
    # ==================================
    action_dim = env.action_spec_unbatched.shape[-1]
    input_shape = env.observation_spec["observation"].shape

    def _safe_gain(act_name: str):
        name = (act_name or "").lower()
        if name in {"leakyrelu", "leaky_relu"}:
            try:
                return torch.nn.init.calculate_gain("leaky_relu", 0.01)
            except Exception:  # noqa: BLE001
                return None
        valid = {
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
            "sigmoid",
            "tanh",
            "relu",
            "leaky_relu",
            "selu",
        }
        if name in valid:
            try:
                return torch.nn.init.calculate_gain(name)
            except Exception:  # noqa: BLE001
                return None
        return None

    # ---- Policy recurrent backbone (LSTM over observations) ----
    actor_lstm = LSTMModule(
        input_size=input_shape[-1],
        hidden_size=hp.lstm_hidden_size_policy,
        num_layers=hp.lstm_num_layers_policy,
        dropout=hp.lstm_dropout if hp.lstm_num_layers_policy > 1 else 0.0,
        in_key="observation",
        out_key="actor_features",
        device=device,
    )

    actor_mlp = MLP(
        in_features=hp.lstm_hidden_size_policy,
        out_features=action_dim if hp.state_ind_normal_scale else 2 * action_dim,
        depth=hp.num_hidden_layers_policy,
        num_cells=hp.num_cells_policy,
        activation_class=getattr(nn, hp.activation_policy),
        device=device,
    )

    actor_gain = _safe_gain(hp.activation_policy)
    if actor_gain is None:
        print(
            "Info: Using PyTorch default initialization for actor MLP because "
            f"activation '{hp.activation_policy}' has no supported gain."
        )
    else:
        for layer in actor_mlp.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=actor_gain)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    if hp.state_ind_normal_scale:
        actor_head = nn.Sequential(
            actor_mlp,
            AddStateIndependentNormalScale(
                action_dim,
                scale_lb=1e-8,
            ).to(device),
        )
    else:
        actor_head = nn.Sequential(
            actor_mlp,
            NormalParamExtractor(
                scale_mapping="biased_softplus_1.0",
                scale_lb=0.1,
            ).to(device),
        )

    actor_head_module = TensorDictModule(
        actor_head,
        in_keys=["actor_features"],
        out_keys=["loc", "scale"],
    ).to(device)

    actor_module = TensorDictSequential(
        actor_lstm,
        actor_head_module,
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

    # ---- Value recurrent backbone ----
    value_lstm = LSTMModule(
        input_size=input_shape[-1],
        hidden_size=hp.lstm_hidden_size_value,
        num_layers=hp.lstm_num_layers_value,
        dropout=hp.lstm_dropout if hp.lstm_num_layers_value > 1 else 0.0,
        in_key="observation",
        out_key="value_features",
        device=device,
    )

    value_mlp = MLP(
        in_features=hp.lstm_hidden_size_value,
        out_features=1,
        depth=hp.num_hidden_layers_value,
        num_cells=hp.num_cells_value,
        activation_class=getattr(nn, hp.activation_value),
        device=device,
    )

    value_gain = _safe_gain(hp.activation_value)
    if value_gain is None:
        print(
            "Info: Using PyTorch default initialization for value MLP because "
            f"activation '{hp.activation_value}' has no supported gain."
        )
    else:
        for layer in value_mlp.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=value_gain)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    value_head = ValueOperator(
        module=value_mlp,
        in_keys=["value_features"],
    ).to(device)

    value_module = TensorDictSequential(
        value_lstm,
        value_head,
    ).to(device)

    # ================================
    # Register primers for recurrence
    # ================================
    # The primers inject recurrent state keys ("hidden", "cell", "is_init", etc.)
    # into the env rollout TensorDict so SliceSampler can later recover them.
    def _append_primers_for_module(env_obj, module):
        primers = get_primers_from_module(module)
        if primers is None:
            return
        if isinstance(primers, TensorDictPrimer):
            env_obj.append_transform(primers)
            return
        try:
            for primer in primers:
                env_obj.append_transform(primer)
        except TypeError:
            env_obj.append_transform(primers)

    _append_primers_for_module(env, policy.module)
    _append_primers_for_module(env, value_module)

    # ================================
    # PPO loss + advantage estimator
    # ================================
    advantage_module = GAE(
        gamma=hp.gamma,
        lmbda=hp.lmbda,
        value_network=value_module,
        average_gae=True,
        deactivate_vmap=True,
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

    optim = torch.optim.Adam(loss_module.parameters(), hp.lr)

    # ====================================
    # Collector across available backends
    # ====================================
    num_devices = get_device_count(backend_name)
    print(f"\nFound {num_devices} devices for backend '{backend_name}'")

    def make_env(backend, device_id):
        """Create environment with specified backend and device ID."""
        env_local = PyFREnvironment(
            mesh_file=mesh_file,
            cfg_file=cfg_file,
            backend_name=backend,
            device_id=device_id,
            ic_dir=ic_dir,
            print_diagnostic=False,
        )
        env_local = TransformedEnv(env_local, Compose(StepCounter(), InitTracker()))
        _append_primers_for_module(env_local, policy.module)
        _append_primers_for_module(env_local, value_module)
        return env_local

    env_makers = [
        (lambda idx=i: make_env(backend_name, idx)) for i in range(num_devices)
    ]

    collector = MultiSyncDataCollector(
        create_env_fn=env_makers,
        policy=policy,
        frames_per_batch=hp.frames_per_batch,
        total_frames=hp.total_frames,
        split_trajs=False,          # keep trajectories as contiguous streams
        reset_at_each_iter=True,    # reset each worker every collector iteration
        device=device,
    )

    # ====================================
    # Replay buffer for recurrent PPO
    # ====================================
    # We are doing on-policy PPO. We still use a replay buffer here as a
    # convenient "minibatch slicer": SliceSampler draws short contiguous
    # sequence windows (length seq_len) from the most recent rollout.
    #
    # batch_size = seqs_per_minibatch  (number of sequences per optimizer step)
    # slice_len  = seq_len             (truncated BPTT length)
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(hp.frames_per_batch, device=device),
        sampler=SliceSampler(
            slice_len=hp.seq_len,
            end_key=("next", "done"),
            cache_values=True,
            strict_length=False,
        ),
        batch_size=hp.seqs_per_minibatch,
    )

    # ====================================
    # Bookkeeping / checkpoint management
    # ====================================
    best_eval_reward = float("-inf")
    best_eval_episode = 0
    start_episode = 0
    current_eval_reward = None
    start_batch_idx = 0

    if load_model and os.path.exists(load_model):
        checkpoint = torch.load(load_model, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        value_module.load_state_dict(checkpoint["value_state_dict"])

        current_eval_reward = checkpoint.get("current_reward", float("-inf"))
        loaded_best_reward = checkpoint.get("best_reward", float("-inf"))
        start_episode = checkpoint.get("episode", 0)
        loaded_best_episode = checkpoint.get("best_episode", 0)
        start_batch_idx = checkpoint.get("batch_idx", 0) + 1

        saved_hp = checkpoint.get("hyperparameters", {})
        differences = []

        if saved_hp:
            print("\nVerifying hyperparameters...")
            for key, saved_value in saved_hp.items():
                if hasattr(hp, key):
                    current_value = getattr(hp, key)
                    if current_value != saved_value:
                        differences.append((key, saved_value, current_value))

        if differences:
            key_width = 22
            val_width = 20
            key_sep = "─" * (key_width + 2)
            val_sep = "─" * (val_width + 2)
            print(
                "\nWARNING: Hyperparameter differences detected between checkpoint and current settings:"
            )
            print(f"┌{key_sep}┬{val_sep}┬{val_sep}┐")
            print(
                f"│ {'Key':<{key_width}} │ {'Checkpoint Value':<{val_width}} │ {'Current Value':<{val_width}} │"
            )
            print(f"├{key_sep}┼{val_sep}┼{val_sep}┤")
            for key, saved, current in differences:
                print(
                    f"│ {key:<{key_width}} │ {str(saved):<{val_width}} │ {str(current):<{val_width}} │"
                )
            print(f"└{key_sep}┴{val_sep}┴{val_sep}┘")
        else:
            print("done.")

        if "config_content" in checkpoint and config_content:
            print("\nVerifying config files...")
            config_differences = compare_configs(
                checkpoint["config_content"], config_content
            )
            if config_differences:
                print(
                    "\nWARNING: Config file differences detected between checkpoint and current:"
                )
                for line_num, ckpt_line, curr_line in config_differences:
                    print(f"Line {line_num}:")
                    print(f"  Checkpoint: {ckpt_line}")
                    print(f"  Current:    {curr_line}")
                    print()
            else:
                print("Config files match between checkpoint and current settings.")

        if (
            hasattr(hp, "print_config_on_load")
            and hp.print_config_on_load
            and "config_content" in checkpoint
        ):
            print("\n=== CHECKPOINT CONFIG FILE CONTENT ===\n")
            print(checkpoint["config_content"])
            print("\n=======================================\n")

        print(f"\nLoaded model from: {load_model}")
        print(f"Current eval reward: {current_eval_reward:.4f}")
        print(f"Best eval reward from checkpoint: {loaded_best_reward:.4f}")
        print(f"Best reward achieved at episode: {loaded_best_episode}")
        print(f"Continuing from episode: {start_episode}\n")

        if isinstance(loaded_best_reward, (int, float)) and (
            loaded_best_reward > best_eval_reward
        ):
            best_eval_reward = loaded_best_reward
            best_eval_episode = loaded_best_episode
            print(f"Updated best reward tracking to: {best_eval_reward:.4f}\n")

    # =========================
    # Logging setup
    # =========================
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best-model.pt")
    latest_model_path = os.path.join(checkpoint_dir, "latest-model.pt")
    logs = defaultdict(list)

    remaining_episodes = hp.episodes - start_episode
    pbar = tqdm(total=remaining_episodes, desc="Training", initial=start_episode)
    episode_count = start_episode

    eval_str = ""

    try:
        episode_pbar = tqdm(total=hp.episodes, desc="Episodes", leave=False)
        env.set_progress_bar(episode_pbar)
    except Exception:  # noqa: BLE001
        print("Warning: Could not create episode progress bar")
        episode_pbar = None
        env.set_progress_bar(None)

    wallclock_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(checkpoint_dir, f"tensorboard_logs/{wallclock_datetime}")
    writer = SummaryWriter(log_dir=log_path)

    # Store hyperparameters to tensorboard
    hparam_dict = {}
    metric_dict = {}
    for key, value in hp.__dict__.items():
        if key not in ["_param_sources", "_derived_params"]:
            if isinstance(value, (int, float, str, bool)) or value is None:
                hparam_dict[key] = value

    run_name = os.path.join(
        os.path.dirname(os.path.realpath(log_path)), f"{wallclock_datetime}"
    )
    print(f"Writing hyperparameters to tensorboard: {run_name}")
    writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)

    # =========================
    # Main training loop
    # =========================
    batch_idx = start_batch_idx
    for _, tensordict_data_cpu in enumerate(collector):
        # on-policy batch collected from all envs
        episode_count += hp.episodes_per_batch

        tensordict_data = tensordict_data_cpu.to(device)

        # quick scalar reward logging from rollout
        train_reward = tensordict_data["next", "reward"].mean().item()
        writer.add_scalar("batch/train_reward", train_reward, batch_idx)
        writer.add_scalar("batch/episodes", episode_count, batch_idx)
        writer.add_scalar(
            "batch/learning_rate", optim.param_groups[0]["lr"], batch_idx
        )

        # For recurrent PPO:
        # We will run hp.num_epochs epochs.
        # In each epoch we:
        #   1. recompute GAE (advantages and value targets) with current critic
        #   2. dump the (updated) rollout into the replay buffer
        #   3. sample hp.updates_per_epoch minibatches, each minibatch is
        #      seqs_per_minibatch sequences of length seq_len, and do an optimizer step
        for epoch_idx in range(hp.num_epochs):
            advantage_module(tensordict_data) # could recompute advantages per innermost loop, like in non-recurrent PPO?

            data_view = tensordict_data.reshape(-1)

            # load rollout into buffer for this epoch
            replay_buffer.extend(data_view)

            for update_idx in range(hp.updates_per_epoch):
                # sample one recurrent minibatch
                subdata = replay_buffer.sample()

                # forward PPO loss dict
                loss_vals = loss_module(subdata)

                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"]
                if hp.entropy_eps > 0:
                    entropy_term = loss_vals.get("loss_entropy", 0.0)
                    if isinstance(entropy_term, torch.Tensor):
                        loss_value = loss_value + entropy_term
                    else:
                        loss_value = loss_value + torch.tensor(
                            entropy_term, device=device
                        )

                # scalars for logging
                policy_obj = loss_vals["loss_objective"].item()
                val_loss = loss_vals["loss_critic"].item()
                ent_loss = (
                    loss_vals.get(
                        "loss_entropy", torch.tensor(0.0, device=device)
                    )
                    .detach()
                    .item()
                )

                # truncated BPTT backward through seq_len
                loss_value.backward()

                grad_norm = nn.utils.clip_grad_norm_(
                    loss_module.parameters(), hp.max_grad_norm
                )

                optim.step()
                optim.zero_grad()

                global_update_idx = (
                    batch_idx * (hp.num_epochs * hp.updates_per_epoch)
                    + epoch_idx * hp.updates_per_epoch
                    + update_idx
                )

                writer.add_scalar(
                    "loss/policy_objective", policy_obj, global_update_idx
                )
                writer.add_scalar("loss/value_loss", val_loss, global_update_idx)
                writer.add_scalar("loss/entropy_bonus", ent_loss, global_update_idx)
                writer.add_scalar("grad/norm", grad_norm, global_update_idx)

        # After PPO updates on this batch, sync new weights to collectors
        collector.update_policy_weights_()

        logs["train_reward"].append(train_reward)

        # Periodic evaluation without exploration noise
        if batch_idx % hp.eval_frequency == 0:
            eval_reward = evaluate_policy(env, policy)
            logs["eval_reward"].append(eval_reward)

            writer.add_scalar("eval/mean_reward", eval_reward, batch_idx + 1)
            writer.add_scalar(
                "train/learning_rate", optim.param_groups[0]["lr"], batch_idx + 1
            )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_episode = episode_count
                print(
                    f"\nNew best eval reward: {best_eval_reward:.5f} at episode {episode_count}"
                )
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "value_state_dict": value_module.state_dict(),
                        "current_reward": eval_reward,
                        "best_reward": best_eval_reward,
                        "episode": episode_count,
                        "best_episode": best_eval_episode,
                        "batch_idx": batch_idx,
                        "hyperparameters": {
                            k: v
                            for k, v in hp.__dict__.items()
                            if not k.startswith("_") and not callable(v)
                        },
                        "config_content": config_content,
                        "config_path": cfg_path,
                    },
                    best_model_path,
                )

            checkpoint_payload = {
                "policy_state_dict": policy.state_dict(),
                "value_state_dict": value_module.state_dict(),
                "current_reward": eval_reward,
                "best_reward": best_eval_reward,
                "episode": episode_count,
                "best_episode": best_eval_episode,
                "batch_idx": batch_idx,
                "hyperparameters": {
                    k: v
                    for k, v in hp.__dict__.items()
                    if not k.startswith("_") and not callable(v)
                },
                "config_content": config_content,
                "config_path": cfg_path,
            }
            torch.save(
                checkpoint_payload,
                os.path.join(checkpoint_dir, f"model-{batch_idx + 1}.pt"),
            )
            torch.save(checkpoint_payload, latest_model_path)

            eval_str = f"eval reward: {eval_reward:.5f} (best: {best_eval_reward:.5f})"

        pbar.set_postfix(
            {
                "train_reward": f"{train_reward:.5f}",
                "eval": eval_str,
                "lr": f"{optim.param_groups[0]['lr']:.2e}",
            }
        )
        pbar.update(hp.episodes_per_batch)

        batch_idx += 1

    pbar.close()
    if episode_pbar:
        episode_pbar.close()

    collector.shutdown()
    writer.close()
    env.close()


def evaluate_policy(env, policy, num_steps=1_000_000):
    """Evaluate policy without exploration using consistent IC"""
    env.set_evaluation_mode(True)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(num_steps, policy)
            eval_reward = eval_rollout["next", "reward"].mean().item()
            del eval_rollout
            return eval_reward
    finally:
        env.set_evaluation_mode(False)


def get_closest_divisor(n, target):
    """
    Finds the closest divisor of n to the target value.
    """
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)

    closest = min(
        divisors,
        key=lambda x: (abs(x - target), -x),
    )
    return closest


def compare_configs(checkpoint_config, current_config):
    """
    Compare two config files line by line and return differences.
    """
    if not checkpoint_config or not current_config:
        return []

    checkpoint_lines = [line.strip() for line in checkpoint_config.splitlines()]
    current_lines = [line.strip() for line in current_config.splitlines()]

    differences = []

    for i, (ckpt_line, curr_line) in enumerate(zip(checkpoint_lines, current_lines)):
        if (
            not ckpt_line
            or ckpt_line.startswith(";")
            or not curr_line
            or curr_line.startswith(";")
        ):
            continue

        if ckpt_line != curr_line:
            differences.append((i + 1, ckpt_line, curr_line))

    if len(checkpoint_lines) > len(current_lines):
        for i, line in enumerate(
            checkpoint_lines[len(current_lines):], start=len(current_lines)
        ):
            if line and not line.startswith(";"):
                differences.append((i + 1, line, "[MISSING]"))

    elif len(current_lines) > len(checkpoint_lines):
        for i, line in enumerate(
            current_lines[len(checkpoint_lines):], start=len(checkpoint_lines)
        ):
            if line and not line.startswith(";"):
                differences.append((i + 1, "[MISSING]", line))

    return differences


@dataclass
class HyperParameters:
    # General settings
    torch_device: str = "cpu"
    print_config_on_load: bool = False

    # Network architecture
    num_hidden_layers_policy: int = 2
    num_hidden_layers_value: int = 2
    num_cells_policy: int = 512
    num_cells_value: int = 512
    activation_policy: str = "Tanh"
    activation_value: str = "Tanh"
    state_ind_normal_scale: bool = False

    # Recurrent settings
    lstm_hidden_size_policy: int = 256
    lstm_hidden_size_value: int = 256
    lstm_num_layers_policy: int = 1
    lstm_num_layers_value: int = 1
    lstm_dropout: float = 0.0

    # PPO / rollout scheduling
    episodes: int = 1200                # total episodes to train
    episodes_per_batch: int = 20        # how many episodes (across envs) collected before one PPO update phase
    num_epochs: int = 4                 # PPO epochs per on-policy batch

    # Recurrent minibatch (truncated BPTT) schedule
    seq_len: int = 64                   # unroll length for LSTM per sampled sequence (TBPTT horizon)
    seqs_per_minibatch: int = 32        # number of sequences per optimizer step
    updates_per_epoch: int = 8          # how many optimizer steps per epoch

    # PPO parameters
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.97
    entropy_eps: float = 1e-3
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    # Evaluation
    eval_frequency: int = 1

    def __post_init__(self):
        self._param_sources = {
            field_name: "default" for field_name in self.__dataclass_fields__.keys()
        }
        # Derived values are computed after env is known
        self._derived_params = {
            "frames_per_batch",
            "total_frames",
            "actions_per_episode",
        }
        self.actions_per_episode = None
        self.frames_per_batch = None
        self.total_frames = None

    def _calculate_derived(self, env):
        # env.dtend / env.action_interval is how many actions per episode
        self.actions_per_episode = int(env.dtend / env.action_interval)

        # frames collected from all envs before PPO update
        # episodes_per_batch episodes * actions_per_episode timesteps each
        self.frames_per_batch = self.episodes_per_batch * self.actions_per_episode

        # total frames seen over entire training horizon
        self.total_frames = self.episodes * self.actions_per_episode

    @classmethod
    def from_config(cls, cfg: Inifile) -> "HyperParameters":
        """
        Load hyperparameters from the 'neuralnetwork-hyperparameters' section
        of the .ini file when present. Fields not present fall back to defaults.
        """
        params = cls()
        if "neuralnetwork-hyperparameters" in cfg.sections():
            section = "neuralnetwork-hyperparameters"
            for field_name, field in params.__dataclass_fields__.items():
                config_name = field_name.replace("_", "-")
                if (
                    field_name not in params._derived_params
                    and cfg.hasopt(section, config_name)
                ):
                    # type dispatch
                    if field.type == int:
                        value = cfg.getint(section, config_name)
                    elif field.type == float:
                        value = cfg.getfloat(section, config_name)
                    elif field.type == bool:
                        value = cfg.getbool(section, config_name)
                    else:
                        value = cfg.get(section, config_name)
                    setattr(params, field_name, value)
                    params._param_sources[field_name] = "config"
        return params

    def print_summary(self) -> None:
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
            "Recurrent Architecture": [
                ("lstm_hidden_size_policy", "Hidden size of policy LSTM"),
                ("lstm_hidden_size_value", "Hidden size of value LSTM"),
                ("lstm_num_layers_policy", "Number of LSTM layers in policy"),
                ("lstm_num_layers_value", "Number of LSTM layers in value"),
                ("lstm_dropout", "Dropout prob between stacked LSTM layers"),
            ],
            "Rollout / PPO Schedule": [
                ("episodes", "Total training episodes"),
                ("episodes_per_batch", "Episodes collected per PPO batch"),
                ("num_epochs", "PPO epochs per on-policy batch"),
                ("seq_len", "Truncated BPTT length (timesteps per sequence)"),
                ("seqs_per_minibatch", "Sequences per optimizer step"),
                ("updates_per_epoch", "Optimizer steps per epoch"),
            ],
            "PPO Parameters": [
                ("clip_epsilon", "PPO clipping parameter"),
                ("gamma", "Discount factor"),
                ("lmbda", "GAE lambda parameter"),
                ("entropy_eps", "Entropy bonus coefficient"),
                ("lr", "Learning rate"),
                ("max_grad_norm", "Gradient clipping norm"),
            ],
            "Derived Values": [
                ("frames_per_batch", "Frames per PPO batch"),
                ("total_frames", "Total frames overall"),
                ("actions_per_episode", "Actions per episode"),
            ],
            "Evaluation Settings": [
                ("eval_frequency", "Evaluate policy every N PPO batches"),
            ],
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
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)
            if current_line:
                lines.append(" ".join(current_line))
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
            f"{tl_corner}{h_line * (param_width + 2)}{t_down}"
            f"{h_line * (value_width + 2)}{t_down}"
            f"{h_line * (src_width + 2)}{t_down}"
            f"{h_line * (desc_width + 2)}{tr_corner}"
        )
        mid_line = (
            f"{t_right}{h_line * (param_width + 2)}{cross}"
            f"{h_line * (value_width + 2)}{cross}"
            f"{h_line * (src_width + 2)}{cross}"
            f"{h_line * (desc_width + 2)}{t_left}"
        )
        bot_line = (
            f"{bl_corner}{h_line * (param_width + 2)}{t_up}"
            f"{h_line * (value_width + 2)}{t_up}"
            f"{h_line * (src_width + 2)}{t_up}"
            f"{h_line * (desc_width + 2)}{br_corner}"
        )

        def format_row(param, value, source, desc_line, is_continuation=False):
            src_display = "" if is_continuation else source
            return (
                f"{v_line} {param:<{param_width}} {v_line} "
                f"{str(value):<{value_width}} {v_line} "
                f"{src_display:<{src_width}} {v_line} "
                f"{desc_line:<{desc_width}} {v_line}"
            )

        def format_header():
            return (
                f"{v_line} {'Parameter':<{param_width}} {v_line} "
                f"{'Value':<{value_width}} {v_line} "
                f"{'Src':<{src_width}} {v_line} "
                f"{'Description':<{desc_width}} {v_line}"
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
                elif (
                    param_name in self._param_sources
                    and self._param_sources[param_name] == "config"
                ):
                    source = "[C]"
                else:
                    source = "[D]"

                desc_lines = wrap_text(description, desc_width)

                print(format_row(param_name, value, source, desc_lines[0]))

                for line in desc_lines[1:]:
                    print(format_row("", "", "", line, is_continuation=True))

            print(bot_line)

        print("\nSource: [C]=From .ini config file, [D]=Default, [-]=Derived")


def get_device_count(backend_name):
    """Get number of available devices for given backend."""
    if backend_name == "hip":
        from pyfr.backends.hip.driver import HIP
        return HIP().device_count()
    if backend_name == "cuda":
        from pyfr.backends.cuda.driver import CUDA
        return CUDA().device_count()
    return 1
