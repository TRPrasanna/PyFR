import os
import time

import matplotlib.pyplot as plt
import numpy as np

from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.rl.algorithms import (
    get_algorithm_spec,
    is_recurrent_algorithm,
    load_model as load_algorithm_model,
    normalize_algorithm_name,
)
from pyfr.rl.core import (
    HyperParameters,
    _load_metadata,
    _resolve_model_path,
    compare_configs,
)
from pyfr.rl.env import (
    CollectiveEnvController,
    PyFREnvironment,
    serve_collective_envs,
    stop_collective_workers,
)


def _resolve_eval_algorithm(cli_algorithm: str | None,
                            metadata: dict | None,
                            hp: HyperParameters | None = None) -> str:
    selected = normalize_algorithm_name(
        cli_algorithm or (hp.algorithm if hp is not None else 'ppo')
    )

    if metadata and metadata.get('algorithm'):
        ckpt_algorithm = normalize_algorithm_name(metadata['algorithm'])
        if cli_algorithm and selected != ckpt_algorithm:
            raise ValueError(
                "Checkpoint algorithm mismatch: "
                f"'--algorithm {selected}' requested, "
                f"but checkpoint uses '{ckpt_algorithm}'."
            )
        selected = ckpt_algorithm

    return selected


def evaluate_policy(mesh_file, cfg_file, backend_name, load_model,
                    ic_dir=None, episodes=1, algorithm=None):
    """Evaluate a trained SB3 policy."""
    init_mpi()
    comm, rank, root = get_comm_rank_root()
    is_root = rank == root
    collective_mode = comm.size > 1

    if hasattr(cfg_file, 'name'):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    try:
        with open(cfg_path, 'r') as f:
            config_content = f.read()
    except Exception as e:
        if is_root:
            print(f'Warning: Could not read config file: {e}')
        config_content = None

    model_path = _resolve_model_path(load_model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {load_model}')

    if model_path.endswith('.pt'):
        raise ValueError(
            'TorchRL .pt checkpoints are not compatible with SB3 evaluation.'
        )

    metadata = _load_metadata(model_path)
    if metadata and config_content and metadata.get('config_content'):
        diffs = compare_configs(metadata['config_content'], config_content)
        if is_root:
            print('\nVerifying config files...')
            if diffs:
                print('\nWARNING: Config file differences detected:')
                for line_num, ckpt_line, curr_line in diffs:
                    print(f'Line {line_num}:')
                    print(f'  Checkpoint: {ckpt_line}')
                    print(f'  Current:    {curr_line}')
            else:
                print('Config files match.')

    eval_device_id = (
        'local-rank' if collective_mode and backend_name in {'cuda', 'hip'} else 0
    )

    raw_env = None
    env = None
    workers_active = False
    try:
        raw_env = PyFREnvironment(
            mesh_file=mesh_file,
            cfg_file=cfg_path,
            backend_name=backend_name,
            device_id=eval_device_id,
            ic_dir=ic_dir,
            print_diagnostic=is_root
        )

        raw_env.set_evaluation_mode(True)

        if collective_mode and not is_root:
            serve_collective_envs({'eval': raw_env})
            return

        workers_active = collective_mode and is_root
        env = CollectiveEnvController('eval', raw_env) if collective_mode else raw_env

        # Print hyperparameter summary (checkpoint preferred, else config).
        hp = None
        if metadata and isinstance(metadata.get('hyperparameters'), dict):
            if is_root:
                print('\nUsing hyperparameters from checkpoint')
            hp = HyperParameters()
            for key, value in metadata['hyperparameters'].items():
                if hasattr(hp, key):
                    setattr(hp, key, value)
        elif 'neuralnetwork-hyperparameters' in raw_env.cfg.sections():
            if is_root:
                print('\nUsing hyperparameters from config file')
            hp = HyperParameters.from_config(raw_env.cfg, announce=is_root)

        selected_algorithm = _resolve_eval_algorithm(algorithm, metadata, hp)
        if hp is not None:
            hp.algorithm = selected_algorithm

        algo_spec = get_algorithm_spec(selected_algorithm)

        if is_root:
            print(f'Loading SB3 model ({algo_spec.display_name}): {model_path}')
        model = load_algorithm_model(
            selected_algorithm,
            model_path,
            env=None,
            device='cpu'
        )

        if hp is not None:
            hp._calculate_derived(raw_env, num_envs=1, announce=is_root)
            if is_root:
                hp.print_summary(num_devices=comm.size if collective_mode else 1,
                                 num_envs=1)

        recurrent = is_recurrent_algorithm(selected_algorithm)

        all_episode_returns = []
        first_ep_actions = None
        first_ep_rewards = None

        for ep in range(episodes):
            obs, _ = env.reset()
            done = False

            ep_actions = []
            ep_rewards = []

            lstm_states = None
            episode_starts = np.array([True], dtype=bool)

            while not done:
                if recurrent:
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True
                    )
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, _ = env.step(action)

                ep_actions.append(np.asarray(action, dtype=np.float64).reshape(-1))
                ep_rewards.append(float(reward))

                done = bool(terminated or truncated)
                if recurrent:
                    episode_starts[0] = done

            ep_rewards_arr = np.asarray(ep_rewards, dtype=np.float64)
            ep_return = float(ep_rewards_arr.sum())
            ep_mean = float(ep_rewards_arr.mean()) if ep_rewards_arr.size else 0.0

            all_episode_returns.append(ep_return)

            if is_root:
                print(
                    f'Episode {ep + 1}/{episodes}: '
                    f'steps={len(ep_rewards)} '
                    f'return={ep_return:.7e} '
                    f'mean_step_reward={ep_mean:.7e}'
                )

            if ep == 0:
                first_ep_actions = np.asarray(ep_actions, dtype=np.float64)
                first_ep_rewards = ep_rewards_arr

        returns = np.asarray(all_episode_returns, dtype=np.float64)
        if is_root:
            print('\nEvaluation Results:')
            print('-' * 40)
            print(f'Episodes: {episodes}')
            print(f'Mean episode return: {returns.mean():.7e}')
            print(f'Std episode return:  {returns.std():.7e}')
            print(f'Min/Max return:      {returns.min():.7e} / {returns.max():.7e}')

            if metadata:
                print('\nCheckpoint Metadata:')
                print('-' * 40)
                if 'algorithm' in metadata:
                    print(f"Algorithm:             {metadata['algorithm']}")
                if 'current_reward' in metadata:
                    print(f"Stored current reward: {metadata['current_reward']:.7e}")
                if 'best_reward' in metadata:
                    print(f"Stored best reward:    {metadata['best_reward']:.7e}")
                if 'best_episode' in metadata:
                    print(f"Best reward episode:   {metadata['best_episode']}")

        if is_root and first_ep_actions is not None and first_ep_rewards is not None:
            num_steps = len(first_ep_rewards)
            num_actions = first_ep_actions.shape[1] if first_ep_actions.ndim == 2 else 1

            if first_ep_actions.ndim == 1:
                first_ep_actions = first_ep_actions.reshape(-1, 1)

            time_array = np.arange(num_steps, dtype=np.float64) * raw_env.action_interval

            print('\nFirst Episode Action History:')
            column_width = 16
            header = f"{'Time':>{column_width}}"
            for i in range(num_actions):
                header += f"{('Action_' + str(i)):>{column_width}}"
            header += f"{'Reward':>{column_width}}"
            print(header)
            print('-' * (column_width * (num_actions + 2)))

            for i in range(num_steps):
                row = f'{time_array[i]:>{column_width}.7e}'
                for j in range(num_actions):
                    row += f'{first_ep_actions[i, j]:>{column_width}.7e}'
                row += f'{first_ep_rewards[i]:>{column_width}.7e}'
                print(row)

            fig, axes = plt.subplots(num_actions + 1, 1,
                                     figsize=(12, 4 * (num_actions + 1)),
                                     sharex=True)
            axes = np.atleast_1d(axes)

            for i in range(num_actions):
                axes[i].plot(time_array, first_ep_actions[:, i], '-',
                             label=f'Action {i}')
                axes[i].set_ylabel(f'Action {i}')
                axes[i].grid(True)
                axes[i].legend()

            axes[-1].plot(time_array, first_ep_rewards, 'r-', label='Reward')
            axes[-1].set_xlabel('Time')
            axes[-1].set_ylabel('Reward')
            axes[-1].grid(True)
            axes[-1].legend()

            plt.tight_layout()
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            plot_filename = f'evaluation_results_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'\nPlot saved as {plot_filename}')

        return float(returns.mean())

    finally:
        if env is not None:
            env.set_evaluation_mode(False)
            env.close()
        elif raw_env is not None:
            raw_env.set_evaluation_mode(False)
            raw_env.close()

        if workers_active:
            stop_collective_workers(comm, root)
