import copy
import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from pyfr.inifile import Inifile
from pyfr.rl.algorithms import (
    create_model as create_algorithm_model,
    get_algorithm_spec,
    is_recurrent_algorithm,
    normalize_algorithm_name,
)
from pyfr.rl.core import (
    HyperParameters,
    SB3OptunaPruningCallback,
    _activation_from_name,
    get_device_count,
)
from pyfr.rl.env import PyFREnvironment


@dataclass
class HPOSettings:
    n_trials: int = 20
    timeout: int | None = None
    study_name: str = 'pyfr-rl-hpo'
    storage: str | None = None
    load_if_exists: bool = True
    direction: str = 'maximize'

    sampler: str = 'tpe'
    n_startup_trials: int = 8
    tpe_multivariate: bool = False
    seed: int = 0

    pruner: str = 'hyperband'
    min_resource: int = 1
    max_resource: int = 6
    reduction_factor: int = 3

    trial_updates: int = 6
    episodes_per_batch: int = 4
    # 0 means auto (one environment per visible backend device).
    envs_per_trial: int = 0
    eval_episodes: int = 1
    save_best_model: bool = True

    tune_params: list[str] | None = None
    device_id: int | None = None

    @classmethod
    def from_config(cls, cfg) -> 'HPOSettings':
        settings = cls()
        section = 'hpo'

        if section not in cfg.sections():
            return settings

        for field_name in settings.__dataclass_fields__:
            key = field_name.replace('_', '-')
            if not cfg.hasopt(section, key):
                continue

            if field_name == 'tune_params':
                value = cfg.getliteral(section, key)
            elif field_name in {'timeout', 'device_id'}:
                value = cfg.getint(section, key)
            elif field_name == 'envs_per_trial':
                raw = cfg.get(section, key).strip().lower()
                value = 0 if raw == 'auto' else int(raw)
            elif field_name in {'study_name', 'storage', 'direction', 'sampler', 'pruner'}:
                value = cfg.get(section, key)
            else:
                current = getattr(settings, field_name)
                if isinstance(current, bool):
                    value = cfg.getbool(section, key)
                elif isinstance(current, int):
                    value = cfg.getint(section, key)
                elif isinstance(current, float):
                    value = cfg.getfloat(section, key)
                else:
                    value = cfg.get(section, key)

            if isinstance(value, str):
                value = value.strip()
                if field_name in {'storage', 'study_name'} and value == '':
                    value = None

            if field_name == 'storage' and value is None:
                setattr(settings, field_name, None)
                continue

            if field_name == 'tune_params' and value is not None and not isinstance(value, (list, tuple)):
                raise ValueError('Invalid hpo setting: tune-params must be a list')

            if field_name == 'tune_params' and value is not None:
                value = list(value)

            if field_name == 'load_if_exists' and isinstance(value, str):
                value = cfg.getbool(section, key)

            setattr(settings, field_name, value)

        if settings.timeout is not None and settings.timeout <= 0:
            settings.timeout = None

        if not settings.storage:
            settings.storage = None

        settings.n_trials = max(1, int(settings.n_trials))
        settings.n_startup_trials = max(0, int(settings.n_startup_trials))
        settings.seed = int(settings.seed)
        settings.min_resource = max(1, int(settings.min_resource))
        settings.max_resource = max(1, int(settings.max_resource))
        settings.reduction_factor = max(2, int(settings.reduction_factor))
        settings.trial_updates = max(1, int(settings.trial_updates))
        settings.episodes_per_batch = max(1, int(settings.episodes_per_batch))
        settings.envs_per_trial = int(settings.envs_per_trial)
        if settings.envs_per_trial < 0:
            settings.envs_per_trial = 0
        settings.eval_episodes = max(1, int(settings.eval_episodes))

        return settings


class _DeriveRef:
    def __init__(self, dtend: float, action_interval: float):
        self.dtend = dtend
        self.action_interval = action_interval


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


def _default_tune_params(algorithm_name: str) -> list[str]:
    base = [
        'lr',
        'clip_epsilon',
        'entropy_eps',
        'gamma',
        'lmbda',
        'num_epochs',
        'desired_num_minibatches',
        'num_cells_policy',
        'num_cells_value',
        'num_hidden_layers_policy',
        'num_hidden_layers_value',
    ]

    if is_recurrent_algorithm(algorithm_name):
        base.extend([
            'lstm_hidden_size',
            'n_lstm_layers',
            'lstm_dropout',
        ])

    return base


def _default_space_spec(param_name: str) -> dict[str, Any] | None:
    specs = {
        'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
        'clip_epsilon': {'type': 'float', 'low': 0.1, 'high': 0.3},
        'entropy_eps': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
        'gamma': {'type': 'float', 'low': 0.95, 'high': 0.999},
        'lmbda': {'type': 'float', 'low': 0.9, 'high': 0.999},
        'num_epochs': {'type': 'int', 'low': 3, 'high': 15, 'step': 1},
        'desired_num_minibatches': {
            'type': 'categorical',
            'choices': [4, 5, 8, 10, 16, 20, 25]
        },
        'num_cells_policy': {
            'type': 'categorical',
            'choices': [128, 256, 384, 512, 768]
        },
        'num_cells_value': {
            'type': 'categorical',
            'choices': [128, 256, 384, 512, 768]
        },
        'num_hidden_layers_policy': {'type': 'int', 'low': 1, 'high': 3, 'step': 1},
        'num_hidden_layers_value': {'type': 'int', 'low': 1, 'high': 3, 'step': 1},
        'lstm_hidden_size': {'type': 'categorical', 'choices': [64, 128, 256, 512]},
        'n_lstm_layers': {'type': 'int', 'low': 1, 'high': 2, 'step': 1},
        'lstm_dropout': {'type': 'float', 'low': 0.0, 'high': 0.2},
    }

    return specs.get(param_name)


def _suggest_from_default_spec(trial, param_name: str, spec: dict[str, Any]):
    ptype = spec['type']
    if ptype == 'categorical':
        return trial.suggest_categorical(param_name, spec['choices'])
    if ptype == 'int':
        return trial.suggest_int(
            param_name,
            int(spec['low']),
            int(spec['high']),
            step=int(spec.get('step', 1))
        )
    if ptype == 'float':
        kwargs = {'log': bool(spec.get('log', False))}
        if 'step' in spec and spec['step'] is not None:
            kwargs['step'] = float(spec['step'])
            kwargs['log'] = False
        return trial.suggest_float(
            param_name,
            float(spec['low']),
            float(spec['high']),
            **kwargs
        )

    raise ValueError(f'Unsupported default spec type: {ptype}')


def _suggest_param_from_config(trial, cfg, hp: HyperParameters,
                               param_name: str, tune_by_default: bool):
    section = 'hpo'
    key = param_name.replace('_', '-')
    current = getattr(hp, param_name)

    fixed_key = f'{key}-fixed'
    choices_key = f'{key}-choices'
    low_key = f'{key}-low'
    high_key = f'{key}-high'
    step_key = f'{key}-step'
    log_key = f'{key}-log'

    if section in cfg.sections() and cfg.hasopt(section, fixed_key):
        try:
            return cfg.getliteral(section, fixed_key)
        except Exception:
            return cfg.get(section, fixed_key)

    if section in cfg.sections() and cfg.hasopt(section, choices_key):
        choices = cfg.getliteral(section, choices_key)
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ValueError(f'Invalid hpo setting: {choices_key} must be a non-empty list')
        return trial.suggest_categorical(param_name, list(choices))

    has_range = (
        section in cfg.sections()
        and cfg.hasopt(section, low_key)
        and cfg.hasopt(section, high_key)
    )
    if has_range:
        if isinstance(current, bool):
            raise ValueError(f'Boolean hyperparameter {param_name} requires choices/fixed, not low/high')

        if isinstance(current, int):
            low = cfg.getint(section, low_key)
            high = cfg.getint(section, high_key)
            step = cfg.getint(section, step_key, 1) if cfg.hasopt(section, step_key) else 1
            return trial.suggest_int(param_name, low, high, step=max(1, step))

        low = cfg.getfloat(section, low_key)
        high = cfg.getfloat(section, high_key)
        log = cfg.getbool(section, log_key) if cfg.hasopt(section, log_key) else False
        step = cfg.getfloat(section, step_key) if cfg.hasopt(section, step_key) else None
        if step is not None:
            log = False
        return trial.suggest_float(param_name, low, high, step=step, log=log)

    if not tune_by_default:
        return current

    spec = _default_space_spec(param_name)
    if spec is None:
        return current

    return _suggest_from_default_spec(trial, param_name, spec)


def _validate_recurrent_hparams(hp: HyperParameters):
    if hp.shared_lstm and hp.enable_critic_lstm:
        raise ValueError(
            "Invalid PPO-LSTM hyperparameters: 'shared-lstm = true' and "
            "'enable-critic-lstm = true' are mutually exclusive."
        )
    if hp.lstm_dropout < 0.0 or hp.lstm_dropout >= 1.0:
        raise ValueError("Invalid PPO-LSTM hyperparameters: 'lstm-dropout' must be in [0, 1).")
    if hp.n_lstm_layers < 1:
        raise ValueError("Invalid PPO-LSTM hyperparameters: 'n-lstm-layers' must be >= 1.")
    if hp.lstm_hidden_size < 1:
        raise ValueError("Invalid PPO-LSTM hyperparameters: 'lstm-hidden-size' must be >= 1.")


def _write_json(path: str, data: dict[str, Any]):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _make_sampler(optuna, settings: HPOSettings):
    sampler = settings.sampler.strip().lower()
    if sampler == 'tpe':
        kwargs = {
            'seed': settings.seed,
            'n_startup_trials': settings.n_startup_trials,
        }
        if settings.tpe_multivariate:
            kwargs['multivariate'] = True
        return optuna.samplers.TPESampler(**kwargs)
    if sampler == 'random':
        return optuna.samplers.RandomSampler(seed=settings.seed)

    raise ValueError(f"Unsupported HPO sampler '{settings.sampler}'. Use 'tpe' or 'random'.")


def _make_pruner(optuna, settings: HPOSettings):
    pruner = settings.pruner.strip().lower()
    if pruner in {'none', 'nop'}:
        return optuna.pruners.NopPruner()
    if pruner == 'hyperband':
        return optuna.pruners.HyperbandPruner(
            min_resource=settings.min_resource,
            max_resource=settings.max_resource,
            reduction_factor=settings.reduction_factor,
        )

    raise ValueError(f"Unsupported HPO pruner '{settings.pruner}'. Use 'hyperband' or 'none'.")


def run_hpo(mesh_file, cfg_file, backend_name, checkpoint_dir='hpo-runs',
            ic_dir=None, algorithm=None, study_name=None, storage=None,
            n_trials=None, timeout=None, sampler=None, pruner=None,
            device_id=None, envs_per_trial=None, episodes_per_batch=None,
            trial_updates=None):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for 'pyfr-rl hpo'. Install it with: "
            "'pip install optuna' (or add it to your environment)."
        ) from exc

    cfg_path = cfg_file.name if hasattr(cfg_file, 'name') else cfg_file
    probe_env = PyFREnvironment(
        mesh_file=mesh_file,
        cfg_file=cfg_path,
        backend_name=backend_name,
        device_id=0,
        ic_dir=ic_dir,
        print_diagnostic=True
    )

    cfg_ini = Inifile.load(cfg_path)
    base_hp = HyperParameters.from_config(cfg_ini)
    selected_algorithm = normalize_algorithm_name(algorithm or base_hp.algorithm)
    base_hp.algorithm = selected_algorithm

    settings = HPOSettings.from_config(cfg_ini)
    if study_name is not None:
        settings.study_name = study_name
    if storage is not None:
        settings.storage = storage
    if n_trials is not None:
        settings.n_trials = max(1, int(n_trials))
    if timeout is not None:
        settings.timeout = int(timeout) if int(timeout) > 0 else None
    if sampler is not None:
        settings.sampler = sampler
    if pruner is not None:
        settings.pruner = pruner
    if device_id is not None:
        settings.device_id = int(device_id)
    if envs_per_trial is not None:
        settings.envs_per_trial = int(envs_per_trial)
        if settings.envs_per_trial < 0:
            settings.envs_per_trial = 0
    if episodes_per_batch is not None:
        settings.episodes_per_batch = max(1, int(episodes_per_batch))
    if trial_updates is not None:
        settings.trial_updates = max(1, int(trial_updates))

    if settings.max_resource > settings.trial_updates:
        print(
            'Note: reducing hpo.max_resource to match trial_updates '
            f'({settings.trial_updates}).'
        )
        settings.max_resource = settings.trial_updates

    if settings.max_resource < settings.min_resource:
        settings.max_resource = settings.min_resource

    # Default persistent study DB lives alongside other HPO artifacts.
    if settings.storage is None:
        default_db = os.path.abspath(os.path.join(checkpoint_dir, 'hpo.db'))
        settings.storage = f'sqlite:///{default_db}'
        print(f'Note: no HPO storage configured; using {settings.storage}')

    visible_devices = max(1, get_device_count(backend_name))
    if settings.envs_per_trial <= 0:
        settings.envs_per_trial = visible_devices
        print(
            'Note: envs-per-trial=auto -> '
            f'using {settings.envs_per_trial} env(s) from detected devices.'
        )

    derive_ref = _DeriveRef(probe_env.dtend, probe_env.action_interval)
    probe_env.close()

    base_hp.episodes_per_batch = settings.episodes_per_batch
    base_hp.episodes = settings.trial_updates * settings.episodes_per_batch
    base_hp._calculate_derived(derive_ref, settings.envs_per_trial)

    tune_params = settings.tune_params or _default_tune_params(selected_algorithm)
    tune_params = [p.strip().replace('-', '_') for p in tune_params if str(p).strip()]
    recurrent_only = {'lstm_hidden_size', 'n_lstm_layers', 'shared_lstm', 'enable_critic_lstm', 'lstm_dropout'}
    if not is_recurrent_algorithm(selected_algorithm):
        dropped = [p for p in tune_params if p in recurrent_only]
        tune_params = [p for p in tune_params if p not in recurrent_only]
        if dropped:
            print(
                'Note: ignoring recurrent-only tune params for PPO: '
                + ', '.join(dropped)
            )

    algo_spec = get_algorithm_spec(selected_algorithm)

    print('\nHPO Configuration')
    print('-' * 60)
    print(f'Algorithm:          {algo_spec.display_name}')
    print(f'Backend:            {backend_name}')
    print(f'Study name:         {settings.study_name}')
    print(f'Storage:            {settings.storage}')
    print(f'Pruner:             {settings.pruner}')
    print(f'Sampler:            {settings.sampler}')
    print(f'Trials:             {settings.n_trials}')
    print(f'Trial updates:      {settings.trial_updates}')
    print(f'Episodes/update:    {settings.episodes_per_batch}')
    print(f'Envs/trial:         {settings.envs_per_trial}')
    print(f'Eval episodes:      {settings.eval_episodes}')
    print(f'Tune params:        {", ".join(tune_params)}')

    os.makedirs(checkpoint_dir, exist_ok=True)
    tb_root = os.path.join(checkpoint_dir, 'tensorboard_logs')
    os.makedirs(tb_root, exist_ok=True)

    def _device_for_env(env_idx: int) -> int:
        if settings.device_id is None:
            return env_idx % visible_devices
        return int(settings.device_id + env_idx) % visible_devices

    sampler_obj = _make_sampler(optuna, settings)
    pruner_obj = _make_pruner(optuna, settings)

    create_kwargs = {
        'direction': settings.direction,
        'sampler': sampler_obj,
        'pruner': pruner_obj,
    }
    if settings.storage:
        create_kwargs['storage'] = settings.storage
        create_kwargs['study_name'] = settings.study_name
        create_kwargs['load_if_exists'] = settings.load_if_exists

    study = optuna.create_study(**create_kwargs)

    def objective(trial):
        trial_hp = copy.deepcopy(base_hp)
        trial_hp.episodes_per_batch = settings.episodes_per_batch
        trial_hp.episodes = settings.trial_updates * settings.episodes_per_batch

        for param_name in tune_params:
            if not hasattr(trial_hp, param_name):
                raise ValueError(f'Unknown hyperparameter for tuning: {param_name}')

            value = _suggest_param_from_config(
                trial,
                cfg=cfg_ini,
                hp=trial_hp,
                param_name=param_name,
                tune_by_default=True
            )
            setattr(trial_hp, param_name, value)
            trial_hp._param_sources[param_name] = 'hpo'

        if not trial_hp.use_sde:
            trial_hp.squash_output = False

        if is_recurrent_algorithm(selected_algorithm):
            _validate_recurrent_hparams(trial_hp)

        trial_hp._calculate_derived(derive_ref, settings.envs_per_trial)

        trial_dir = os.path.join(checkpoint_dir, f'trial-{trial.number:05d}')
        os.makedirs(trial_dir, exist_ok=True)

        print(
            '[hpo] '
            f'start trial={trial.number} '
            f'params={trial.params}'
        )

        train_env_fns = [
            partial(
                _make_pyfr_env,
                mesh_file=mesh_file,
                cfg_path=cfg_path,
                backend_name=backend_name,
                device_id=_device_for_env(i),
                ic_dir=ic_dir,
                print_diagnostic=False,
                evaluation_mode=False
            )
            for i in range(settings.envs_per_trial)
        ]

        train_env = None
        eval_env = None
        try:
            if settings.envs_per_trial == 1:
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
                    device_id=_device_for_env(0),
                    ic_dir=ic_dir,
                    print_diagnostic=False,
                    evaluation_mode=True
                )
            ])
            eval_env = VecMonitor(eval_env)

            policy_kwargs = {
                'activation_fn': _activation_from_name(trial_hp.activation_policy),
                'squash_output': trial_hp.squash_output,
                'net_arch': {
                    'pi': [trial_hp.num_cells_policy] * trial_hp.num_hidden_layers_policy,
                    'vf': [trial_hp.num_cells_value] * trial_hp.num_hidden_layers_value,
                }
            }

            if is_recurrent_algorithm(selected_algorithm):
                policy_kwargs.update({
                    'lstm_hidden_size': trial_hp.lstm_hidden_size,
                    'n_lstm_layers': trial_hp.n_lstm_layers,
                    'shared_lstm': trial_hp.shared_lstm,
                    'enable_critic_lstm': trial_hp.enable_critic_lstm,
                })
                if trial_hp.lstm_dropout > 0.0:
                    policy_kwargs['lstm_kwargs'] = {'dropout': trial_hp.lstm_dropout}

            model = create_algorithm_model(
                selected_algorithm,
                env=train_env,
                hp=trial_hp,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_root,
            )

            callback = SB3OptunaPruningCallback(
                trial=trial,
                eval_env=eval_env,
                hp=trial_hp,
                recurrent_policy=is_recurrent_algorithm(selected_algorithm),
                n_eval_episodes=settings.eval_episodes,
                best_model_path=(
                    os.path.join(trial_dir, 'best-model.zip')
                    if settings.save_best_model else None
                ),
                verbose=1
            )

            model.learn(
                total_timesteps=trial_hp.total_frames,
                callback=callback,
                progress_bar=False,
                reset_num_timesteps=True,
                tb_log_name=f'trial-{trial.number:05d}'
            )

            objective_value = callback.best_eval_reward
            if objective_value == float('-inf'):
                objective_value = (
                    float(callback.latest_eval_reward)
                    if callback.latest_eval_reward is not None else float('-inf')
                )

            trial_record = {
                'trial': trial.number,
                'state': 'pruned' if callback.pruned else 'complete',
                'value': float(objective_value),
                'latest_eval_reward': (
                    None if callback.latest_eval_reward is None else float(callback.latest_eval_reward)
                ),
                'best_eval_reward': float(callback.best_eval_reward),
                'params': trial.params,
                'resolved_hyperparameters': asdict(trial_hp),
            }
            _write_json(os.path.join(trial_dir, 'trial-summary.json'), trial_record)

            if callback.pruned:
                raise optuna.TrialPruned(f'Pruned at update {callback.pruned_update}')

            return float(objective_value)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            fail_record = {
                'trial': trial.number,
                'state': 'failed',
                'error': str(exc),
                'params': trial.params,
            }
            _write_json(os.path.join(trial_dir, 'trial-summary.json'), fail_record)
            raise
        finally:
            if train_env is not None:
                train_env.close()
            if eval_env is not None:
                eval_env.close()

    study.optimize(
        objective,
        n_trials=settings.n_trials,
        timeout=settings.timeout,
        gc_after_trial=True,
        show_progress_bar=False
    )

    print('\nHPO Results')
    print('-' * 60)
    print(f'Total trials: {len(study.trials)}')
    states = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    for s, c in sorted(states.items()):
        print(f'{s:>10}: {c}')

    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    if best_trial is not None:
        print('\nBest Trial')
        print(f'number: {best_trial.number}')
        print(f'value:  {study.best_value:.7e}')
        print('params:')
        for k, v in best_trial.params.items():
            print(f'  {k}: {v}')

        best_out = {
            'best_trial_number': int(best_trial.number),
            'best_value': float(study.best_value),
            'best_params': best_trial.params,
            'study_name': settings.study_name,
            'storage': settings.storage,
            'algorithm': selected_algorithm,
        }
        _write_json(os.path.join(checkpoint_dir, 'best-trial.json'), best_out)

    return study
