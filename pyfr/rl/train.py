from functools import partial
import os
import time

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from pyfr.rl.algorithms import (
    create_model as create_algorithm_model,
    get_algorithm_spec,
    is_recurrent_algorithm,
    load_model as load_algorithm_model,
    normalize_algorithm_name,
)
from pyfr.rl.core import (
    HyperParameters,
    SB3EvalAndCheckpointCallback,
    _activation_from_name,
    _load_metadata,
    _resolve_model_path,
    compare_configs,
    get_device_count,
)
from pyfr.rl.env import PyFREnvironment


def _make_pyfr_env(mesh_file, cfg_path, backend_name, device_id,
                   ic_dir=None, print_diagnostic=False, evaluation_mode=False):
    env = PyFREnvironment(
        mesh_file=mesh_file,
        cfg_file=cfg_path,
        backend_name=backend_name,
        device_id=device_id,
        ic_dir=ic_dir,
        print_diagnostic=print_diagnostic
    )

    if evaluation_mode:
        env.set_evaluation_mode(True)

    return env


def _resolve_selected_algorithm(hp: HyperParameters, cli_algorithm: str | None,
                                metadata: dict | None) -> str:
    selected = normalize_algorithm_name(cli_algorithm or hp.algorithm)

    if metadata and metadata.get('algorithm'):
        ckpt_algorithm = normalize_algorithm_name(metadata['algorithm'])
        if cli_algorithm and ckpt_algorithm != selected:
            raise ValueError(
                "Checkpoint algorithm mismatch: "
                f"'--algorithm {selected}' requested, "
                f"but checkpoint uses '{ckpt_algorithm}'."
            )

        if ckpt_algorithm != selected:
            print(
                'Note: overriding configured algorithm '
                f"'{selected}' with checkpoint algorithm '{ckpt_algorithm}'."
            )
        selected = ckpt_algorithm

    return selected


def train_agent(mesh_file, cfg_file, backend_name,
                checkpoint_dir='checkpoints', ic_dir=None, load_model=None,
                algorithm=None):
    if hasattr(cfg_file, 'name'):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    try:
        with open(cfg_path, 'r') as f:
            config_content = f.read()
    except Exception as e:
        print(f'Warning: Could not read config file: {e}')
        config_content = None

    # Probe one environment for diagnostics and to compute derived params.
    probe_env = PyFREnvironment(
        mesh_file, cfg_path, backend_name, 0,
        ic_dir=ic_dir, print_diagnostic=True
    )

    if 'neuralnetwork-hyperparameters' not in probe_env.cfg.sections():
        print('No neuralnetwork-hyperparameters section found. Using defaults.')

    hp = HyperParameters.from_config(probe_env.cfg)

    resolved_model_path = None
    metadata = None
    start_timesteps = 0
    start_episode = 0
    best_reward = float('-inf')

    if load_model:
        resolved_model_path = _resolve_model_path(load_model)
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(f'Model file not found: {load_model}')

        if resolved_model_path.endswith('.pt'):
            raise ValueError(
                'TorchRL .pt checkpoints are not loadable by SB3. '
                'Please start a new SB3 training run or convert the model.'
            )

        metadata = _load_metadata(resolved_model_path)

    selected_algorithm = _resolve_selected_algorithm(hp, algorithm, metadata)
    hp.algorithm = selected_algorithm

    if hp.squash_output and not hp.use_sde:
        # Keep backward compatibility for existing configs that do not set
        # SB3 gSDE options: default squash_output=True is only meaningful with
        # use_sde=True. If squash-output was explicitly requested in config,
        # raise; otherwise disable it automatically.
        squash_from_cfg = hp._param_sources.get('squash_output') == 'config'
        if squash_from_cfg:
            raise ValueError(
                "Invalid hyperparameter combination: 'squash-output = true' "
                "requires 'use-sde = true' in SB3."
            )

        print(
            "Note: default 'squash-output = true' is disabled because "
            "'use-sde = false'. Set 'use-sde = true' to enable squashing."
        )
        hp.squash_output = False

    if is_recurrent_algorithm(selected_algorithm):
        if hp.shared_lstm and hp.enable_critic_lstm:
            raise ValueError(
                "Invalid PPO-LSTM hyperparameters: 'shared-lstm = true' and "
                "'enable-critic-lstm = true' are mutually exclusive."
            )

        if hp.lstm_dropout < 0.0 or hp.lstm_dropout >= 1.0:
            raise ValueError(
                "Invalid PPO-LSTM hyperparameters: 'lstm-dropout' must be in "
                '[0, 1).'
            )

        if hp.n_lstm_layers < 1:
            raise ValueError(
                "Invalid PPO-LSTM hyperparameters: 'n-lstm-layers' must be >= 1."
            )

        if hp.lstm_hidden_size < 1:
            raise ValueError(
                "Invalid PPO-LSTM hyperparameters: 'lstm-hidden-size' must be >= 1."
            )

    if metadata is not None:
        start_timesteps = int(metadata.get('timesteps', 0))
        start_episode = int(metadata.get('episode', 0))
        best_reward = float(metadata.get('best_reward', float('-inf')))

        if config_content and metadata.get('config_content'):
            print('\nVerifying config files...')
            cfg_diffs = compare_configs(metadata['config_content'],
                                        config_content)
            if cfg_diffs:
                print('\nWARNING: Config differences detected:')
                for line_num, ckpt_line, curr_line in cfg_diffs:
                    print(f'Line {line_num}:')
                    print(f'  Checkpoint: {ckpt_line}')
                    print(f'  Current:    {curr_line}')
            else:
                print('Config files match.')

        if hp.print_config_on_load and metadata.get('config_content'):
            print('\n=== CHECKPOINT CONFIG FILE CONTENT ===\n')
            print(metadata['config_content'])
            print('\n=======================================\n')

    num_devices = max(1, get_device_count(backend_name))
    num_envs = max(1, num_devices * max(1, hp.envs_per_device))

    hp._calculate_derived(probe_env, num_envs)
    hp.print_summary(num_devices=num_devices, num_envs=num_envs)

    probe_env.close()

    algo_spec = get_algorithm_spec(selected_algorithm)

    print(f"\nFound {num_devices} devices for backend '{backend_name}'")
    print(
        f"Using SB3 ({algo_spec.display_name}) "
        f'with {num_envs} environment(s).'
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    tb_log_root = os.path.join(checkpoint_dir, 'tensorboard_logs')
    os.makedirs(tb_log_root, exist_ok=True)

    train_env_fns = [
        partial(
            _make_pyfr_env,
            mesh_file=mesh_file,
            cfg_path=cfg_path,
            backend_name=backend_name,
            device_id=(i % num_devices),
            ic_dir=ic_dir,
            print_diagnostic=False,
            evaluation_mode=False
        )
        for i in range(num_envs)
    ]

    if num_envs == 1:
        train_env = DummyVecEnv(train_env_fns)
    else:
        train_env = SubprocVecEnv(train_env_fns, start_method='spawn')
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([
        partial(
            _make_pyfr_env,
            mesh_file=mesh_file,
            cfg_path=cfg_path,
            backend_name=backend_name,
            device_id=0,
            ic_dir=ic_dir,
            print_diagnostic=False,
            evaluation_mode=True
        )
    ])
    eval_env = VecMonitor(eval_env)

    policy_kwargs = {
        'activation_fn': _activation_from_name(hp.activation_policy),
        'squash_output': hp.squash_output,
        'net_arch': {
            'pi': [hp.num_cells_policy] * hp.num_hidden_layers_policy,
            'vf': [hp.num_cells_value] * hp.num_hidden_layers_value,
        }
    }

    if is_recurrent_algorithm(selected_algorithm):
        policy_kwargs.update({
            'lstm_hidden_size': hp.lstm_hidden_size,
            'n_lstm_layers': hp.n_lstm_layers,
            'shared_lstm': hp.shared_lstm,
            'enable_critic_lstm': hp.enable_critic_lstm,
        })

        if hp.lstm_dropout > 0.0:
            policy_kwargs['lstm_kwargs'] = {'dropout': hp.lstm_dropout}

    model = None

    if load_model:
        print(
            f"Loading SB3 model ({algo_spec.display_name}): "
            f'{resolved_model_path}'
        )
        model = load_algorithm_model(
            selected_algorithm,
            resolved_model_path,
            env=train_env,
            device=hp.torch_device,
        )
        model.verbose = 0

    if model is None:
        model = create_algorithm_model(
            selected_algorithm,
            env=train_env,
            hp=hp,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_root,
        )

    remaining_timesteps = hp.total_frames - start_timesteps
    if remaining_timesteps <= 0:
        print(
            'No remaining timesteps to train: '
            f'total_frames={hp.total_frames}, start_timesteps={start_timesteps}'
        )
        train_env.close()
        eval_env.close()
        return

    wallclock_datetime = time.strftime('%Y-%m-%d_%H-%M-%S')

    callback = SB3EvalAndCheckpointCallback(
        eval_env=eval_env,
        hp=hp,
        checkpoint_dir=checkpoint_dir,
        cfg_path=cfg_path,
        config_content=config_content,
        algorithm_name=selected_algorithm,
        recurrent_policy=is_recurrent_algorithm(selected_algorithm),
        start_episode=start_episode,
        start_timesteps=start_timesteps,
        best_eval_reward=best_reward,
        verbose=1,
    )

    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callback,
        reset_num_timesteps=(start_timesteps == 0),
        tb_log_name=wallclock_datetime,
        progress_bar=False,
    )

    train_env.close()
    eval_env.close()
