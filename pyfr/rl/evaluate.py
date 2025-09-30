# evaluate_lstm.py
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor, TanhNormal, NormalParamExtractor, MLP, LSTMModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import TransformedEnv, Compose, StepCounter, InitTracker

from .train import HyperParameters, compare_configs  # reuse your dataclass + config diff
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.rl.env import PyFREnvironment


def evaluate_policy_lstm(mesh_file, cfg_file, backend_name, load_model, ic_dir=None, episodes=1):
    """
    Evaluate a PPO-LSTM policy: rebuilds LSTM actor with the same keys as training,
    appends the primer, loads the checkpoint, runs a deterministic rollout, and plots.
    """
    # Resolve cfg path and read for comparison
    if hasattr(cfg_file, "name"):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    try:
        with open(cfg_path, "r") as f:
            config_content = f.read()
    except Exception as e:
        print(f"Warning: Could not read config file: {e}")
        config_content = None

    # Env with StepCounter + InitTracker (same as training)
    env = PyFREnvironment(
        mesh_file, cfg_path, backend_name, device_id=0, ic_dir=ic_dir, print_diagnostic=True
    )
    env = TransformedEnv(env, Compose(StepCounter(), InitTracker()))

    # Load checkpoint
    if not os.path.exists(load_model):
        print(f"Error: Model file not found: {load_model}")
        sys.exit(1)
    device = torch.device("cpu")  # evaluate on CPU by default
    checkpoint = torch.load(load_model, map_location=device)

    # Hyperparameters: use config since PPO-LSTM checkpoints may not store hparams
    if "neuralnetwork-hyperparameters" not in env.cfg.sections():
        print("No [neuralnetwork-hyperparameters] in cfg. Using defaults.")
    hp = HyperParameters.from_config(env.cfg)
    hp._calculate_derived(env)

    # Config diff if available
    if "config_content" in checkpoint and config_content:
        print("\nVerifying config files...")
        diffs = compare_configs(checkpoint["config_content"], config_content)
        if diffs:
            print("\nWARNING: Config file differences detected between checkpoint and current:")
            for line_num, ckpt_line, curr_line in diffs:
                print(f"Line {line_num}:")
                print(f"  Checkpoint: {ckpt_line}")
                print(f"  Current:    {curr_line}\n")
        else:
            print("Config files match between checkpoint and current settings.")

    if getattr(hp, "print_config_on_load", False) and "config_content" in checkpoint:
        print("\n=== CHECKPOINT CONFIG FILE CONTENT ===\n")
        print(checkpoint["config_content"])
        print("\n=======================================\n")

    # Build actor LSTM + head exactly like training
    action_dim = env.action_spec_unbatched.shape[-1]
    obs_dim = env.observation_spec["observation"].shape[-1]

    # LSTM with distinct hidden keys
    actor_lstm = LSTMModule(
        input_size=obs_dim,
        hidden_size=hp.num_cells_policy,                  # LSTM hidden size (actor)
        device=device,
        in_keys=["observation", "actor_h", "actor_c"],
        out_keys=["actor_feat", ("next", "actor_h"), ("next", "actor_c")],
        python_based=True,
    )

    # Primer must be appended to env so rollout carries recurrent states
    env.append_transform(actor_lstm.make_tensordict_primer())

    # Head MLP on top of LSTM features → params → (loc, scale)
    actor_head = MLP(
        in_features=hp.num_cells_policy,
        out_features=2 * action_dim,
        num_cells=[hp.num_cells_policy],
        activation_class=getattr(nn, hp.activation_policy),
        device=device,
    )
    actor_head_mod = TensorDictModule(actor_head, in_keys=["actor_feat"], out_keys=["actor_params"])

    param_extract = TensorDictModule(
        NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=0.1),
        in_keys=["actor_params"],
        out_keys=["loc", "scale"],
    )

    actor_net = TensorDictSequential(actor_lstm, actor_head_mod, param_extract).to(device)

    policy = ProbabilisticActor(
        module=actor_net,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=False,  # eval only
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
            "tanh_loc": False,
        },
    ).to(device)

    # Load weights
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    # Pull stored scalar metadata if present
    current_reward = checkpoint.get("current_reward", checkpoint.get("reward", None))
    best_reward = checkpoint.get("best_reward", current_reward)
    saved_episode = checkpoint.get("episode", 0)
    best_episode = checkpoint.get("best_episode", saved_episode)
    batch_idx = checkpoint.get("batch_idx", None)

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

    # Network summary
    print("\nNetwork Architecture (LSTM policy):")
    print("-" * 40)
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"LSTM hidden (actor): {hp.num_cells_policy}")
    print(f"Head activation: {hp.activation_policy}")

    # Deterministic evaluation
    env.set_evaluation_mode(True)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            print("\nStarting evaluation...")
            # keep a large cap like your PPO eval
            eval_rollout = env.rollout(100000, policy)

            actions = eval_rollout["action"].cpu().numpy()
            rewards = eval_rollout["next", "reward"].cpu().numpy().flatten()

            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            num_actions = actions.shape[1]
            time_array = np.arange(len(actions)) * env.action_interval

            # Print action history
            print("\nAction history:")
            colw = 16
            header = f"{'Time':>{colw}}"
            for i in range(num_actions):
                header += f"{('Action_'+str(i)):>{colw}}"
            header += f"{'Reward':>{colw}}"
            print(header)
            print("-" * (colw * (num_actions + 2)))
            for t in range(len(time_array)):
                row = f"{time_array[t]:>{colw}.7e}"
                for i in range(num_actions):
                    row += f"{actions[t, i]:>{colw}.7e}"
                row += f"{rewards[t]:>{colw}.7e}"
                print(row)

            # Stats
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
            if current_reward is not None and current_reward != 0.0:
                print(f"Difference from expected: {((eval_reward - current_reward)/current_reward)*100:.2f}%")

            # Plots
            fig, axes = plt.subplots(num_actions + 1, 1, figsize=(12, 4 * (num_actions + 1)), sharex=True)
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

            ts = time.strftime("%Y%m%d-%H%M%S")
            plot_filename = f"evaluation_results_lstm_{ts}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\nPlots saved as {plot_filename}")

            del eval_rollout
            return eval_reward

    except Exception as e:
        print(f"Unexpected error in LSTM evaluation: {str(e)}")
        raise
    finally:
        env.set_evaluation_mode(False)
