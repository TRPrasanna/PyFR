import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import (
    AddStateIndependentNormalScale,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl.envs import Compose, InitTracker, StepCounter, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.modules import (
    LSTMModule,
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.modules.utils import get_primers_from_module

from .train import HyperParameters, compare_configs
from pyfr.rl.env import PyFREnvironment


def _append_primers(env_obj, module):
    """Attach recurrent primers to an environment for a given module."""
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


def evaluate_policy(
    mesh_file,
    cfg_file,
    backend_name,
    load_model,
    ic_dir=None,
    episodes=1,
):
    """Evaluate a trained PPO-LSTM policy."""
    if hasattr(cfg_file, "name"):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    try:
        with open(cfg_path, "r") as f:
            config_content = f.read()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Could not read config file: {exc}")
        config_content = None

    env = PyFREnvironment(
        mesh_file,
        cfg_path,
        backend_name,
        device_id=0,
        ic_dir=ic_dir,
        print_diagnostic=True,
    )
    env = TransformedEnv(env, Compose(StepCounter(), InitTracker()))

    if not os.path.exists(load_model):
        print(f"Error: Model file not found: {load_model}")
        sys.exit(1)

    checkpoint = torch.load(load_model, map_location="cpu")

    if "hyperparameters" in checkpoint:
        print("Using hyperparameters from checkpoint")
        hp = HyperParameters()
        for key, value in checkpoint["hyperparameters"].items():
            if hasattr(hp, key):
                setattr(hp, key, value)
        hp._calculate_derived(env)
    else:
        print("No hyperparameters in checkpoint, using values from config file")
        if "neuralnetwork-hyperparameters" not in env.cfg.sections():
            print(
                "No neuralnetwork-hyperparameters section found in config file. "
                "Using default hyperparameters."
            )
        hp = HyperParameters.from_config(env.cfg)
        hp._calculate_derived(env)

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

    action_dim = env.action_spec_unbatched.shape[-1]
    input_shape = env.observation_spec["observation"].shape
    device = torch.device(hp.torch_device)

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

    actor_module = TensorDictSequential(actor_lstm, actor_head_module).to(device)
    policy = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=False,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
            "tanh_loc": False,
        },
    ).to(device)

    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    _append_primers(env, policy.module)

    current_reward = checkpoint.get("current_reward", checkpoint.get("reward"))
    best_reward = checkpoint.get("best_reward", current_reward)
    saved_episode = checkpoint.get("episode", 0)
    best_episode = checkpoint.get("best_episode", saved_episode)
    batch_idx = checkpoint.get("batch_idx")

    print("\nModel Information:")
    print("-" * 40)
    if current_reward is not None:
        print(f"Current reward: {current_reward:.4f}")
    if best_reward is not None:
        print(f"Best reward: {best_reward:.4f}")
        print(f"Best reward at episode: {best_episode}")
    print(f"Model saved at episode: {saved_episode}")
    if batch_idx is not None:
        print(f"Model saved at batch: {batch_idx}")
    print(f"Model path: {load_model}")

    print("\nNetwork Architecture:")
    print("-" * 40)
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {action_dim}")
    print(f"LSTM hidden size: {hp.lstm_hidden_size_policy}")
    print(f"LSTM layers: {hp.lstm_num_layers_policy}")
    print(f"MLP hidden layers: {hp.num_hidden_layers_policy}")
    print(f"MLP hidden units: {hp.num_cells_policy}")
    print(f"Activation: {hp.activation_policy}")
    print(f"State-independent normal scale: {hp.state_ind_normal_scale}")

    env.set_evaluation_mode(True)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            print("\nStarting evaluation...")
            eval_rollout = env.rollout(100000, policy)

            actions = eval_rollout["action"].cpu().numpy()
            rewards = eval_rollout["next", "reward"].cpu().numpy().flatten()

            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            num_actions = actions.shape[1]
            time_array = np.arange(len(actions)) * env.action_interval

            print("\nAction history:")
            column_width = 16
            header = f"{'Time':>{column_width}}"
            for i in range(num_actions):
                header += f"{('Action_'+str(i)):>{column_width}}"
            header += f"{'Reward':>{column_width}}"
            print(header)
            print("-" * (column_width * (num_actions + 2)))

            for t_idx in range(len(time_array)):
                row = f"{time_array[t_idx]:>{column_width}.7e}"
                for i in range(num_actions):
                    row += f"{actions[t_idx, i]:>{column_width}.7e}"
                row += f"{rewards[t_idx]:>{column_width}.7e}"
                print(row)

            eval_reward = float(np.mean(rewards))
            eval_std = float(np.std(rewards))
            eval_min = float(np.min(rewards))
            eval_max = float(np.max(rewards))
            eval_total = float(np.sum(rewards))

            print("\nEvaluation Results:")
            print("-" * 40)
            if current_reward is not None:
                print(f"Expected reward: {current_reward:.4f}")
            print(f"Actual mean reward: {eval_reward:.4f}")
            print(f"Reward std dev: {eval_std:.4f}")
            print(f"Min/Max rewards: {eval_min:.4f} / {eval_max:.4f}")
            print(f"Total reward: {eval_total:.4f}")
            print(f"Number of steps: {len(rewards)}")

            if current_reward:
                diff_pct = (eval_reward - current_reward) / current_reward * 100
                print(f"Difference from expected: {diff_pct:.2f}%")

            fig, axes = plt.subplots(
                num_actions + 1,
                1,
                figsize=(12, 4 * (num_actions + 1)),
                sharex=True,
            )
            axes = np.atleast_1d(axes)

            for i in range(num_actions):
                axes[i].plot(time_array, actions[:, i], "-", label=f"Action {i}")
                axes[i].set_ylabel(f"Action {i}")
                axes[i].grid(True)
                axes[i].legend()

            axes[-1].plot(time_array, rewards, "r-", label="Reward")
            axes[-1].set_xlabel("Time")
            axes[-1].set_ylabel("Reward")
            axes[-1].grid(True)
            axes[-1].legend()

            plt.tight_layout()

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plot_filename = f"evaluation_results_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"\nPlots saved as {plot_filename}")
            del eval_rollout
            return eval_reward
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error in evaluation: {exc}")
        raise
    finally:
        env.set_evaluation_mode(False)
