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
from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
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
    # Resolve config path (string) for reproducibility in checkpoints
    if hasattr(cfg_file, "name"):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    # Read config file content now so we can store it in checkpoints
    try:
        with open(cfg_path, "r") as f:
            config_content = f.read()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not read config file: {e}")
        config_content = None

    # =========================
    # Environment initialization
    # =========================
    # StepCounter(): gives per-episode step index
    # InitTracker(): marks reset boundaries ("is_init"), which TorchRL uses to reset recurrent state
    env = PyFREnvironment(
        mesh_file, cfg_path, backend_name, 0, ic_dir=ic_dir, print_diagnostic=True
    )
    env = TransformedEnv(env, Compose(StepCounter(), InitTracker()))

    if "neuralnetwork-hyperparameters" not in env.cfg.sections():
        print(
            "No neuralnetwork-hyperparameters section found in config file. "
            "Proceeding to use default hyperparameters."
        )

    hp = HyperParameters.from_config(env.cfg)
    hp._calculate_derived(env)  # fills in actions_per_episode, train_batch_size, total_frames
    device = torch.device(hp.torch_device)

    # =======================================================
    # RLlib-style minibatch / TBPTT schedule resolution
    # =======================================================
    #
    # RLlib knobs we mirror:
    #   - train_batch_size         (derived): total timesteps per update batch
    #   - recurrent_seq_len        (hp.recurrent_seq_len): TBPTT window
    #   - sgd_minibatch_size       (hp.sgd_minibatch_size): timesteps per SGD minibatch
    #   - num_sgd_iter             (hp.num_sgd_iter): epochs over that batch
    #
    # Constraints we enforce:
    #   1. recurrent_seq_len <= episode_length
    #      (otherwise we'd have to slice in the middle and stitch hidden state;
    #       we can add that later, but for now we clamp and warn)
    #
    #   2. sgd_minibatch_size must:
    #        - be >0
    #        - be a multiple of recurrent_seq_len
    #        - divide train_batch_size with no remainder
    #
    #      If user gives 0 or something invalid, we AUTO-PICK a safe value
    #      and warn. The auto-pick tries to use full episodes as minibatches
    #      (so each minibatch is contiguous and we don't split episodes).
    #
    # After this resolution, EVERY timestep in the batch is used exactly once
    # per epoch. No leftovers are dropped.

    episode_len = hp.actions_per_episode  # length of a single rollout episode in env steps
    requested_seq_len = hp.recurrent_seq_len

    if requested_seq_len > episode_len:
        print(
            f"WARNING: requested recurrent_seq_len ({requested_seq_len}) "
            f"is longer than a single episode length ({episode_len}). "
            "This would force mid-episode truncation and hidden-state stitching, "
            "and would normally waste data. "
            f"Clamping recurrent_seq_len down to {episode_len} so that each sequence "
            "is exactly one full episode and we can consume the entire batch."
        )
        effective_seq_len = episode_len
    else:
        effective_seq_len = requested_seq_len

    # train_batch_size is total frames per update, like RLlib train_batch_size
    train_batch_size = hp.train_batch_size

    # Now pick/adjust sgd_minibatch_size
    requested_minibatch = hp.sgd_minibatch_size

    def _valid_minibatch_sizes(train_bs, seq_len):
        # return all divisors of train_bs that are also multiples of seq_len
        vals = []
        for m in range(seq_len, train_bs + 1, seq_len):
            if train_bs % m == 0:
                vals.append(m)
        return vals

    valid_sizes = _valid_minibatch_sizes(train_batch_size, effective_seq_len)

    if not valid_sizes:
        # This should basically never happen for sane integers,
        # but just in case, fallback to using the whole batch.
        valid_sizes = [train_batch_size]

    def _pick_closest_size(target, candidates):
        # choose candidate closest to target; if tie, choose smaller
        return min(candidates, key=lambda x: (abs(x - target), x))

    schedule_notes = []

    if requested_minibatch <= 0:
        # auto mode: prefer "one full episode per minibatch" if that tiles perfectly.
        # one episode per minibatch = episode_len.
        if (episode_len in valid_sizes) and (train_batch_size % episode_len == 0):
            chosen_minibatch = episode_len
        else:
            # fallback: choose the smallest valid size to get more SGD steps
            chosen_minibatch = min(valid_sizes)

        schedule_notes.append(
            "[schedule notice] sgd_minibatch_size was 0 or invalid; "
            f"auto-selected {chosen_minibatch} so that we cover all {train_batch_size} steps with no leftovers."
        )
    else:
        # user-specified minibatch size: snap it to something valid
        chosen_minibatch = _pick_closest_size(requested_minibatch, valid_sizes)
        if chosen_minibatch != requested_minibatch:
            schedule_notes.append(
                "[schedule notice] requested sgd_minibatch_size "
                f"{requested_minibatch} cannot tile train_batch_size={train_batch_size} "
                f"with recurrent_seq_len={effective_seq_len}. "
                f"Using {chosen_minibatch} instead so every sample is used."
            )

    # how many minibatches per epoch
    actual_num_minibatches = train_batch_size // chosen_minibatch

    # sanity: coverage
    used_steps = actual_num_minibatches * chosen_minibatch
    assert used_steps == train_batch_size, (
        "internal bug: minibatch tiling did not cover full batch"
    )

    # summary print
    print("\nRecurrent PPO minibatch schedule (RLlib-style naming):")
    print(f"  train_batch_size           = {train_batch_size}  # total timesteps per update")
    print(f"  requested recurrent_seq_len= {requested_seq_len}")
    print(f"  effective_seq_len          = {effective_seq_len}")
    print(f"  requested sgd_minibatch    = {requested_minibatch}")
    print(f"  chosen sgd_minibatch_size  = {chosen_minibatch}")
    print(f"  num_sgd_iter (epochs)      = {hp.num_sgd_iter}")
    print(f"  minibatches per epoch      = {actual_num_minibatches}")
    print(f"  coverage check: {actual_num_minibatches} * {chosen_minibatch} = {used_steps} (should equal {train_batch_size})")
    for note in schedule_notes:
        print(note)

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

    # Policy recurrent backbone
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

    # Value recurrent backbone
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
    # Primers inject recurrent state keys ("recurrent_state_h", "recurrent_state_c", "is_init", etc.)
    # into the rollout tensordict so SliceSamplerWithoutReplacement can grab contiguous sequences.
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
        frames_per_batch=hp.train_batch_size,  # RLlib: train_batch_size
        total_frames=hp.total_frames,
        split_trajs=False,          # keep trajectories contiguous
        reset_at_each_iter=True,    # reset each worker between collector iterations
        device=device,
    )

    # ====================================
    # Replay buffer for recurrent PPO minibatch slicing
    # ====================================
    # SliceSamplerWithoutReplacement will:
    #   - draw contiguous sequences of length slice_len (= effective_seq_len),
    #   - pack several of those sequences together into a batch whose total length
    #     equals chosen_minibatch.
    # We set batch_size=minibatch_size_steps (chosen_minibatch).
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(hp.train_batch_size, device=device),
        sampler=SliceSamplerWithoutReplacement(
            slice_len=effective_seq_len,
            end_key=("next", "done"),
            strict_length=False,
        ),
        batch_size=chosen_minibatch,
    )

    # ====================================
    # Checkpoint bookkeeping
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
            key_width = 28
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

    remaining_episodes = hp.episodes_total - start_episode
    pbar = tqdm(total=remaining_episodes, desc="Training", initial=start_episode)
    episode_count = start_episode

    eval_str = ""

    try:
        episode_pbar = tqdm(total=hp.episodes_total, desc="Episodes", leave=False)
        env.set_progress_bar(episode_pbar)
    except Exception:
        print("Warning: Could not create episode progress bar")
        episode_pbar = None
        env.set_progress_bar(None)

    wallclock_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(checkpoint_dir, f"tensorboard_logs/{wallclock_datetime}")
    writer = SummaryWriter(log_dir=log_path)

    # Store hyperparameters & resolved schedule to tensorboard
    hparam_dict = {}
    metric_dict = {}
    for key, value in hp.__dict__.items():
        if key not in ["_param_sources", "_derived_params"]:
            if isinstance(value, (int, float, str, bool)) or value is None:
                hparam_dict[key] = value
    # log derived schedule values that matter for debugging
    hparam_dict["effective_seq_len"] = effective_seq_len
    hparam_dict["chosen_sgd_minibatch_size"] = chosen_minibatch
    hparam_dict["actual_num_minibatches"] = actual_num_minibatches
    hparam_dict["train_batch_size"] = train_batch_size

    run_name = os.path.join(
        os.path.dirname(os.path.realpath(log_path)), f"{wallclock_datetime}"
    )
    print(f"Writing hyperparameters to tensorboard: {run_name}")
    writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)

    updates_per_batch = hp.num_sgd_iter * actual_num_minibatches

    # =========================
    # Main training loop
    # =========================
    batch_idx = start_batch_idx
    for _, tensordict_data in enumerate(collector):
        # tensordict_data_cpu: rollout from all env workers,
        # length should be train_batch_size steps total
        episode_count += hp.rollout_fragment_episodes

        # quick scalar reward logging from rollout
        train_reward = tensordict_data["next", "reward"].mean().item()
        writer.add_scalar("batch/train_reward", train_reward, batch_idx)
        writer.add_scalar("batch/episodes_so_far", episode_count, batch_idx)
        writer.add_scalar(
            "batch/learning_rate", optim.param_groups[0]["lr"], batch_idx
        )
        
        for epoch_idx in range(hp.num_sgd_iter):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            # refill replay buffer for this epoch (ring buffer semantics)
            replay_buffer.extend(data_view)

            for sub_update_idx in range(actual_num_minibatches):
                subdata = replay_buffer.sample()

                # PPO loss dict
                loss_vals = loss_module(subdata)
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"]
                if hp.entropy_eps > 0:
                    loss_value = loss_value + loss_vals["loss_entropy"]

                # scalars for logging
                policy_obj = loss_vals["loss_objective"].item()
                val_loss = loss_vals["loss_critic"].item()
                ent_loss = (
                    loss_vals.get("loss_entropy", 0.0).item()
                    if isinstance(loss_vals.get("loss_entropy", 0.0), torch.Tensor)
                    else 0.0
                )

                # truncated BPTT backward through effective_seq_len
                loss_value.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    loss_module.parameters(), hp.max_grad_norm
                )

                global_update_idx = (
                    batch_idx * updates_per_batch
                    + epoch_idx * actual_num_minibatches
                    + sub_update_idx
                )

                writer.add_scalar(
                    "loss/policy_objective", policy_obj, global_update_idx
                )
                writer.add_scalar("loss/value_loss", val_loss, global_update_idx)
                writer.add_scalar("loss/entropy_bonus", ent_loss, global_update_idx)
                writer.add_scalar("grad/norm", grad_norm, global_update_idx)

                optim.step()
                optim.zero_grad()

        # push the new weights to collectors
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

            eval_str = (
                f"eval reward: {eval_reward:.5f} (best: {best_eval_reward:.5f})"
            )
        else:
            eval_str = eval_str

        pbar.set_postfix(
            {
                "train_reward": f"{train_reward:.5f}",
                "eval": eval_str,
                "lr": f"{optim.param_groups[0]['lr']:.2e}",
            }
        )
        pbar.update(hp.rollout_fragment_episodes)

        batch_idx += 1

    pbar.close()
    if episode_pbar:
        episode_pbar.close()

    collector.shutdown()
    writer.close()
    env.close()


def evaluate_policy(env, policy, num_steps=1_000_000):
    """Evaluate policy without exploration"""
    env.set_evaluation_mode(True)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(num_steps, policy)
            eval_reward = eval_rollout["next", "reward"].mean().item()
            del eval_rollout
            return eval_reward
    finally:
        env.set_evaluation_mode(False)


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

    # Recurrent net sizes
    lstm_hidden_size_policy: int = 256
    lstm_hidden_size_value: int = 256
    lstm_num_layers_policy: int = 1
    lstm_num_layers_value: int = 1
    lstm_dropout: float = 0.0

    # === RLlib-style PPO / rollout scheduling ===
    episodes_total: int = 2000            # total number of episodes we plan to run
    rollout_fragment_episodes: int = 20    # how many full episodes we gather per update iteration
    recurrent_seq_len: int = 100          # requested TBPTT window (RLlib: recurrent_seq_len)
    sgd_minibatch_size: int = 0          # requested timesteps per SGD minibatch (RLlib: sgd_minibatch_size). 0=auto
    num_sgd_iter: int = 10                # PPO epochs per batch (RLlib: num_sgd_iter)

    # PPO loss parameters
    clip_epsilon: float = 0.2
    gamma: float = 0.97
    lmbda: float = 0.97
    entropy_eps: float = 1e-3
    lr: float = 1e-4
    max_grad_norm: float = 10.0

    # Eval
    eval_frequency: int = 1

    # ---------- Derived values (filled after env is known) ----------
    # actions_per_episode: timesteps per episode (env.dtend / env.action_interval)
    # train_batch_size:    total timesteps per PPO update iteration
    # total_frames:        total timesteps across entire training horizon
    def __post_init__(self):
        self._param_sources = {
            field_name: "default" for field_name in self.__dataclass_fields__.keys()
        }
        self._derived_params = {
            "actions_per_episode",
            "train_batch_size",
            "total_frames",
        }
        self.actions_per_episode = None
        self.train_batch_size = None
        self.total_frames = None

    def _calculate_derived(self, env):
        # one episode length in env steps
        self.actions_per_episode = int(env.dtend / env.action_interval)

        # RLlib-style train_batch_size = rollout_fragment_episodes * episode_len
        self.train_batch_size = (
            self.rollout_fragment_episodes * self.actions_per_episode
        )

        # total frames for bookkeeping / collector
        self.total_frames = self.episodes_total * self.actions_per_episode

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
        # nice console dump, updated to RLlib-y naming where relevant
        sections = {
            "General Settings": [
                ("torch_device", "'cuda' or 'cpu'"),
                ("print_config_on_load", "Print config file content on model load"),
            ],
            "Network Architecture": [
                ("num_hidden_layers_policy", "Hidden layers in policy MLP"),
                ("num_hidden_layers_value", "Hidden layers in value MLP"),
                ("num_cells_policy", "Hidden units per layer (policy)"),
                ("num_cells_value", "Hidden units per layer (value)"),
                ("activation_policy", "Activation function for policy MLP"),
                ("activation_value", "Activation function for value MLP"),
                ("state_ind_normal_scale", "Use state-independent action std"),
            ],
            "Recurrent Architecture": [
                ("lstm_hidden_size_policy", "Policy LSTM hidden size"),
                ("lstm_hidden_size_value", "Value LSTM hidden size"),
                ("lstm_num_layers_policy", "Policy LSTM layers"),
                ("lstm_num_layers_value", "Value LSTM layers"),
                ("lstm_dropout", "LSTM dropout between stacked layers"),
            ],
            "Rollout / PPO Schedule (RLlib-ish)": [
                ("episodes_total", "Total env episodes to train"),
                ("rollout_fragment_episodes", "Episodes gathered per update"),
                ("recurrent_seq_len", "Requested recurrent_seq_len (TBPTT window)"),
                ("sgd_minibatch_size", "Requested sgd_minibatch_size (timesteps per SGD minibatch, 0=auto)"),
                ("num_sgd_iter", "num_sgd_iter (epochs per batch)"),
            ],
            "PPO Parameters": [
                ("clip_epsilon", "PPO clipping epsilon"),
                ("gamma", "Discount factor"),
                ("lmbda", "GAE lambda"),
                ("entropy_eps", "Entropy bonus coeff"),
                ("lr", "Learning rate"),
                ("max_grad_norm", "Gradient clipping norm"),
            ],
            "Derived Values": [
                ("actions_per_episode", "Episode length in env steps"),
                ("train_batch_size", "train_batch_size (timesteps per update)"),
                ("total_frames", "Total timesteps over full training run"),
            ],
            "Evaluation Settings": [
                ("eval_frequency", "Evaluate policy every N updates"),
            ],
        }

        param_width = 30
        value_width = 15
        src_width = 5
        desc_width = 50

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
