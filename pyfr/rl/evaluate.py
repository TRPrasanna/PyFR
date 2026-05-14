import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, AddStateIndependentNormalScale
from torchrl.modules import (
    ProbabilisticActor, TanhNormal, ValueOperator,
    NormalParamExtractor, MLP,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import StepCounter, TransformedEnv

from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.rl.env import (
    CollectiveEnvController,
    PyFREnvironment,
    serve_collective_envs,
    stop_collective_workers,
)
from .train import (
    HyperParameters, compare_configs, _abort_collective_workers,
    _select_backend_device_id, _tanh_normal_kwargs,
)


def _action_leaf_spec(env):
    spec = env.action_spec_unbatched

    # Depending on the TorchRL version/wrappers, action_spec_unbatched may be
    # either a Composite with an "action" leaf or the Bounded action spec
    # directly.  Only index by key when it is actually a Composite-like spec.
    keys = getattr(spec, 'keys', None)
    if callable(keys):
        try:
            if 'action' in keys(True, True):
                return spec['action']
        except TypeError:
            if 'action' in keys():
                return spec['action']

    return spec


def evaluate_policy(mesh_file, cfg_file, backend_name, load_model,
                    ic_dir=None, episodes=1):
    """Evaluate trained policy."""
    init_mpi()
    comm, rank, root = get_comm_rank_root()
    is_root = rank == root
    collective_mode = comm.size > 1

    # Get config path at the start
    if hasattr(cfg_file, 'name'):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    # Read the config file content for comparison
    try:
        with open(cfg_path, 'r') as f:
            config_content = f.read()
    except Exception as e:
        if is_root:
            print(f"Warning: Could not read config file: {e}")
        config_content = None

    # Determine device-id
    device_id = _select_backend_device_id(collective_mode, backend_name)

    # Initialize environment (MPI-collective)
    raw_env = PyFREnvironment(
        mesh_file, cfg_path, backend_name, device_id=device_id,
        ic_dir=ic_dir, print_diagnostic=is_root
    )
    raw_env.set_evaluation_mode(True)
    base_env = raw_env

    # Non-root ranks: serve and return
    if collective_mode and not is_root:
        serve_collective_envs({'eval': raw_env})
        return

    workers_active = collective_mode
    env = None
    fatal_exc = None

    try:
        if collective_mode:
            raw_env = CollectiveEnvController('eval', raw_env)

        env = TransformedEnv(raw_env, StepCounter())

        # Load model checkpoint
        if not os.path.exists(load_model):
            raise FileNotFoundError(f'Model file not found: {load_model}')

        device = torch.device('cpu')
        checkpoint = torch.load(load_model, map_location=device)

        # First try to use hyperparameters from checkpoint
        if 'hyperparameters' in checkpoint:
            print("Using hyperparameters from checkpoint")
            hp_dict = checkpoint['hyperparameters']
            hp = HyperParameters()
            for key, value in hp_dict.items():
                if hasattr(hp, key):
                    setattr(hp, key, value)
            hp._calculate_derived(base_env)
        else:
            print(
                "No hyperparameters in checkpoint, "
                "using values from config file"
            )
            if 'neuralnetwork-hyperparameters' not in base_env.cfg.sections():
                print(
                    "No neuralnetwork-hyperparameters section found "
                    "in config file. Using default hyperparameters."
                )
            hp = HyperParameters.from_config(base_env.cfg)
            hp._calculate_derived(base_env)

        # Compare config files if both are available
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

        # Actor network with proper output handling
        action_spec = _action_leaf_spec(env)
        action_dim = action_spec.shape[-1]
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

        # Initialize weights for consistency with training
        for layer in actor_mlp.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 1.0)
                layer.bias.data.zero_()

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
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=False,
            distribution_kwargs=_tanh_normal_kwargs(action_spec, device),
        ).to(device)

        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()

        # Get stored rewards and episodes
        current_reward = checkpoint.get(
            'current_reward', checkpoint.get('reward', None)
        )
        best_reward = checkpoint.get('best_reward', current_reward)
        saved_episode = checkpoint.get('episode', 0)
        best_episode = checkpoint.get('best_episode', saved_episode)
        batch_idx = checkpoint.get('batch_idx', None)

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
        print(f"Hidden layers: {hp.num_hidden_layers_policy}")
        print(f"Hidden units: {hp.num_cells_policy}")
        print(f"Activation: {hp.activation_policy}")
        print(
            f"State-independent normal scale: {hp.state_ind_normal_scale}"
        )

        if collective_mode:
            print(
                f"\nMPI collective mode: {comm.size} rank(s)"
            )

        # Run evaluation
        print("\nStarting evaluation...")
        with set_exploration_type(ExplorationType.DETERMINISTIC), \
                torch.no_grad():
            eval_rollout = env.rollout(100000, policy)

            actions = eval_rollout["action"].cpu().numpy()
            rewards = (
                eval_rollout["next", "reward"].cpu().numpy().flatten()
            )

            if len(actions.shape) == 1:
                actions = actions.reshape(-1, 1)
            num_actions = actions.shape[1]
            time_array = (
                np.arange(len(actions)) * base_env.action_interval
            )

            # Print action history
            print("\nAction history:")
            column_width = 16
            header = f"{'Time':>{column_width}}"
            for i in range(num_actions):
                header += f"{('Action_' + str(i)):>{column_width}}"
            header += f"{'Reward':>{column_width}}"
            print(header)
            print("-" * (column_width * (num_actions + 2)))

            for t in range(len(time_array)):
                row = f"{time_array[t]:>{column_width}.7e}"
                for i in range(num_actions):
                    row += f"{actions[t, i]:>{column_width}.7e}"
                row += f"{rewards[t]:>{column_width}.7e}"
                print(row)

            # Calculate statistics
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

            if current_reward is not None:
                pct = ((eval_reward - current_reward)
                       / current_reward) * 100
                print(
                    f"Difference from expected: {pct:.2f}%"
                )

            # Create evaluation plots
            fig, axes = plt.subplots(
                num_actions + 1, 1,
                figsize=(12, 4 * (num_actions + 1)),
                sharex=True
            )
            axes = np.atleast_1d(axes)

            for i in range(num_actions):
                axes[i].plot(
                    time_array, actions[:, i], '-',
                    label=f'Action {i}'
                )
                axes[i].set_ylabel(f'Action {i}')
                axes[i].grid(True)
                axes[i].legend()

            axes[-1].plot(
                time_array, rewards, 'r-', label='Reward'
            )
            axes[-1].set_xlabel('Time')
            axes[-1].set_ylabel('Reward')
            axes[-1].grid(True)
            axes[-1].legend()

            plt.tight_layout()

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plot_filename = f'evaluation_results_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\nPlots saved as {plot_filename}")
            del eval_rollout
            return eval_reward

    except BaseException as exc:
        fatal_exc = exc
        raise
    finally:
        if fatal_exc is not None and workers_active:
            _abort_collective_workers(comm, fatal_exc)
        else:
            try:
                env.set_evaluation_mode(False)
                env.close()
            except Exception:
                try:
                    raw_env.set_evaluation_mode(False)
                    raw_env.close(raise_if_closed=False)
                except Exception:
                    pass

            if workers_active:
                stop_collective_workers(comm, root)
