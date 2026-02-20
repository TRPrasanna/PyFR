from dataclasses import asdict, dataclass
from functools import partial
import json
import math
import os
import time
from typing import Any

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from tqdm.auto import tqdm

from pyfr.inifile import Inifile
from pyfr.rl.env import PyFREnvironment


def _model_meta_path(model_path: str) -> str:
    if model_path.endswith('.zip'):
        return f'{model_path[:-4]}.meta.json'
    return f'{model_path}.meta.json'


def _resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return path

    if os.path.exists(f'{path}.zip'):
        return f'{path}.zip'

    return path


def _save_metadata(model_path: str, metadata: dict[str, Any]) -> None:
    with open(_model_meta_path(model_path), 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _load_metadata(model_path: str) -> dict[str, Any] | None:
    meta_path = _model_meta_path(model_path)
    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


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


def _activation_from_name(name: str):
    if hasattr(nn, name):
        return getattr(nn, name)

    aliases = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'elu': nn.ELU,
        'gelu': nn.GELU,
        'selu': nn.SELU,
        'silu': nn.SiLU,
        'swish': nn.SiLU,
        'leakyrelu': nn.LeakyReLU,
        'leaky_relu': nn.LeakyReLU,
    }

    key = (name or '').lower()
    if key in aliases:
        return aliases[key]

    print(f"Warning: Unknown activation '{name}'. Falling back to Tanh.")
    return nn.Tanh


def compare_configs(checkpoint_config, current_config):
    """
    Compare two config files line by line and return differences.

    Args:
        checkpoint_config: Config content from checkpoint as string
        current_config: Current config content as string

    Returns:
        List of tuples with (line_number, checkpoint_line, current_line)
        for different lines.
    """
    if not checkpoint_config or not current_config:
        return []

    checkpoint_lines = [line.strip() for line in checkpoint_config.splitlines()]
    current_lines = [line.strip() for line in current_config.splitlines()]

    differences = []

    for i, (ckpt_line, curr_line) in enumerate(zip(checkpoint_lines,
                                                   current_lines)):
        if not ckpt_line or ckpt_line.startswith(';'):
            continue
        if not curr_line or curr_line.startswith(';'):
            continue

        if ckpt_line != curr_line:
            differences.append((i + 1, ckpt_line, curr_line))

    if len(checkpoint_lines) > len(current_lines):
        for i, line in enumerate(checkpoint_lines[len(current_lines):],
                                 start=len(current_lines)):
            if line and not line.startswith(';'):
                differences.append((i + 1, line, '[MISSING]'))
    elif len(current_lines) > len(checkpoint_lines):
        for i, line in enumerate(current_lines[len(checkpoint_lines):],
                                 start=len(checkpoint_lines)):
            if line and not line.startswith(';'):
                differences.append((i + 1, '[MISSING]', line))

    return differences


def get_closest_divisor(n, target):
    """Find the closest divisor of n to target."""
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)

    return min(divisors, key=lambda x: (abs(x - target), -x))


def get_device_count(backend_name):
    """Get number of available devices for a backend."""
    if backend_name == 'hip':
        from pyfr.backends.hip.driver import HIP
        return HIP().device_count()
    if backend_name == 'cuda':
        from pyfr.backends.cuda.driver import CUDA
        return CUDA().device_count()

    return 1


@dataclass
class HyperParameters:
    # General settings
    torch_device: str = 'cpu'  # 'cpu', 'cuda', 'auto'
    print_config_on_load: bool = False

    # Network architecture
    num_hidden_layers_policy: int = 2
    num_hidden_layers_value: int = 2
    num_cells_policy: int = 512
    num_cells_value: int = 512
    activation_policy: str = 'Tanh'
    activation_value: str = 'Tanh'

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
    use_sde: bool = False
    sde_sample_freq: int = -1
    squash_output: bool = True

    # Evaluation settings
    eval_frequency: int = 1

    # SB3 worker controls
    envs_per_device: int = 1
    seed: int = 0

    # Derived
    actions_per_episode: int | None = None
    frames_per_batch: int | None = None
    total_frames: int | None = None
    n_steps: int | None = None
    rollout_size: int | None = None
    batch_size: int | None = None

    def __post_init__(self):
        self._param_sources = {
            field_name: 'default'
            for field_name in self.__dataclass_fields__.keys()
        }
        self._derived_params = {
            'actions_per_episode', 'frames_per_batch', 'total_frames',
            'n_steps', 'rollout_size', 'batch_size'
        }

    def _calculate_derived(self, env, num_envs):
        self.actions_per_episode = max(1, int(env.dtend / env.action_interval))
        self.frames_per_batch = self.episodes_per_batch * self.actions_per_episode
        self.total_frames = self.episodes * self.actions_per_episode

        # SB3 rollout size is n_steps * n_envs.
        self.n_steps = max(1, int(round(self.frames_per_batch / num_envs)))
        self.rollout_size = self.n_steps * num_envs

        desired_mb = max(1, self.desired_num_minibatches)
        if self.rollout_size % desired_mb != 0:
            adjusted = get_closest_divisor(self.rollout_size, desired_mb)
            print(
                'Warning: rollout_size '
                f'({self.rollout_size}) is not divisible by '
                f'desired_num_minibatches ({desired_mb}). '
                f'Adjusted desired_num_minibatches to {adjusted}.'
            )
            self.desired_num_minibatches = adjusted

        self.batch_size = max(1, self.rollout_size // self.desired_num_minibatches)

    @classmethod
    def from_config(cls, cfg: Inifile) -> 'HyperParameters':
        params = cls()

        if 'neuralnetwork-hyperparameters' not in cfg.sections():
            return params

        section = 'neuralnetwork-hyperparameters'
        for field_name, field in params.__dataclass_fields__.items():
            config_name = field_name.replace('_', '-')

            if field_name in params._derived_params:
                continue

            if not cfg.hasopt(section, config_name):
                continue

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

        # Legacy TorchRL option; no longer supported in SB3 path.
        if cfg.hasopt(section, 'state-ind-normal-scale'):
            print(
                "Warning: 'state-ind-normal-scale' is deprecated and ignored. "
                "Use 'use-sde = true/false' for SB3 gSDE control."
            )

        return params

    def print_summary(self, num_devices, num_envs):
        sections = {
            'General Settings': [
                ('torch_device', "'cuda', 'cpu', or 'auto'"),
                ('print_config_on_load', 'Print config file content on model load')
            ],
            'Runtime Parallelism': [
                ('num_devices', 'Detected backend devices'),
                ('envs_per_device', 'Environments spawned per detected device'),
                ('num_envs', 'Total environments in vectorized collector'),
                ('seed', 'SB3 random seed')
            ],
            'Network Architecture': [
                ('num_hidden_layers_policy', 'No. of hidden layers in policy network'),
                ('num_hidden_layers_value', 'No. of hidden layers in value network'),
                ('num_cells_policy', 'Size of policy network hidden layers'),
                ('num_cells_value', 'Size of value network hidden layers'),
                ('activation_policy', 'Activation function used by SB3 policy/value'),
                ('activation_value', 'Read from config; SB3 shares activation_fn')
            ],
            'Training Schedule': [
                ('episodes', 'Total training episodes target'),
                ('episodes_per_batch', 'Episodes per policy update target'),
                ('desired_num_minibatches', 'Target minibatches per update'),
                ('num_epochs', 'Training epochs per update')
            ],
            'PPO Parameters': [
                ('clip_epsilon', 'PPO clipping parameter'),
                ('gamma', 'Discount factor'),
                ('lmbda', 'GAE lambda parameter'),
                ('entropy_eps', 'Entropy bonus coefficient'),
                ('lr', 'Learning rate'),
                ('max_grad_norm', 'Gradient clipping norm'),
                ('use_sde', 'Enable generalized State-Dependent Exploration'),
                ('sde_sample_freq', 'Noise resample frequency (-1 = per rollout)'),
                ('squash_output', 'Tanh-squash actions (requires use_sde=true)')
            ],
            'SB3 Derived Values': [
                ('actions_per_episode', 'Actions per episode'),
                ('frames_per_batch', 'Target frames per update'),
                ('total_frames', 'Target total training frames'),
                ('n_steps', 'SB3 rollout steps per environment'),
                ('rollout_size', 'SB3 rollout size = n_steps * num_envs'),
                ('batch_size', 'SB3 minibatch size')
            ],
            'Evaluation Settings': [
                ('eval_frequency', 'Evaluate every N update cycles')
            ]
        }

        runtime_values = {
            'num_devices': num_devices,
            'num_envs': num_envs
        }

        # Define consistent column widths
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

        h_line = '─'
        v_line = '│'
        tl_corner = '┌'
        tr_corner = '┐'
        bl_corner = '└'
        br_corner = '┘'
        t_down = '┬'
        t_up = '┴'
        t_right = '├'
        t_left = '┤'
        cross = '┼'

        top_line = (
            f'{tl_corner}{h_line * (param_width + 2)}{t_down}'
            f'{h_line * (value_width + 2)}{t_down}'
            f'{h_line * (src_width + 2)}{t_down}'
            f'{h_line * (desc_width + 2)}{tr_corner}'
        )
        mid_line = (
            f'{t_right}{h_line * (param_width + 2)}{cross}'
            f'{h_line * (value_width + 2)}{cross}'
            f'{h_line * (src_width + 2)}{cross}'
            f'{h_line * (desc_width + 2)}{t_left}'
        )
        bot_line = (
            f'{bl_corner}{h_line * (param_width + 2)}{t_up}'
            f'{h_line * (value_width + 2)}{t_up}'
            f'{h_line * (src_width + 2)}{t_up}'
            f'{h_line * (desc_width + 2)}{br_corner}'
        )

        def format_row(param, value, source, desc_line, is_continuation=False):
            src_display = '' if is_continuation else source
            return (
                f'{v_line} {param:<{param_width}} {v_line} '
                f'{str(value):<{value_width}} {v_line} '
                f'{src_display:<{src_width}} {v_line} '
                f'{desc_line:<{desc_width}} {v_line}'
            )

        def format_header():
            return (
                f"{v_line} {'Parameter':<{param_width}} {v_line} "
                f"{'Value':<{value_width}} {v_line} "
                f"{'Src':<{src_width}} {v_line} "
                f"{'Description':<{desc_width}} {v_line}"
            )

        print('\nHyperparameters Configuration (SB3)')

        for section_name, params in sections.items():
            print(f'\n{section_name}:')
            print(top_line)
            print(format_header())
            print(mid_line)

            for param_name, description in params:
                if param_name in runtime_values:
                    value = runtime_values[param_name]
                    source = '[-]'
                else:
                    value = getattr(self, param_name)
                    if param_name in self._derived_params:
                        source = '[-]'
                    elif (
                        param_name in self._param_sources and
                        self._param_sources[param_name] == 'config'
                    ):
                        source = '[C]'
                    else:
                        source = '[D]'

                desc_lines = wrap_text(description, desc_width)
                print(format_row(param_name, value, source, desc_lines[0]))
                for line in desc_lines[1:]:
                    print(format_row('', '', '', line, is_continuation=True))

            print(bot_line)

        print('\nSource: [C]=From .ini config file, [D]=Default, [-]=Derived/runtime')

        if self.activation_policy != self.activation_value:
            print(
                '\nNote: SB3 uses a shared activation_fn for policy and value. '
                f"Using activation_policy='{self.activation_policy}'."
            )

        if self.squash_output and not self.use_sde:
            print(
                '\nNote: squash_output=true requires use_sde=true in SB3.'
            )


class SB3EvalAndCheckpointCallback(BaseCallback):
    def __init__(self, eval_env, hp: HyperParameters, checkpoint_dir: str,
                 cfg_path: str, config_content: str | None,
                 start_episode: int = 0,
                 start_timesteps: int = 0,
                 best_eval_reward: float = float('-inf'),
                 verbose: int = 1):
        super().__init__(verbose=verbose)

        self.eval_env = eval_env
        self.hp = hp
        self.checkpoint_dir = checkpoint_dir
        self.cfg_path = cfg_path
        self.config_content = config_content
        self.start_episode = start_episode
        self.start_timesteps = start_timesteps

        self.best_eval_reward = best_eval_reward
        self.latest_eval_reward = None
        self.best_eval_episode = start_episode

        self.eval_freq_steps = max(1, hp.eval_frequency * hp.rollout_size)
        self._last_eval_timestep = 0

        self.best_model_path = os.path.join(checkpoint_dir, 'best-model.zip')
        self.latest_model_path = os.path.join(checkpoint_dir, 'latest-model.zip')

        # TorchRL-like progress reporting state
        self._pbar = None
        self._episodes_shown = max(0, int(start_episode))
        self._latest_train_reward = None
        self._latest_eval_str = ''

    def _current_lr(self) -> float:
        try:
            return float(self.model.policy.optimizer.param_groups[0]['lr'])
        except Exception:
            return 0.0

    def _episodes_done(self) -> int:
        return int(self._total_timesteps() / max(1, self.hp.actions_per_episode))

    def _set_progress_postfix(self):
        if self._pbar is None:
            return

        postfix = {}
        if self._latest_train_reward is not None:
            postfix['train_reward'] = f'{self._latest_train_reward:.5f}'
        if self._latest_eval_str:
            postfix['eval'] = self._latest_eval_str
        postfix['lr'] = f'{self._current_lr():.2e}'

        self._pbar.set_postfix(postfix)

    def _sync_progress_bar(self):
        if self._pbar is None:
            return

        done = min(self.hp.episodes, self._episodes_done())
        if done > self._episodes_shown:
            self._pbar.update(done - self._episodes_shown)
            self._episodes_shown = done

    def _total_timesteps(self) -> int:
        # When resuming, some SB3 saves restore num_timesteps and some flows
        # restart it from zero. Make metadata monotonic in both cases.
        cur = int(self.model.num_timesteps)
        if cur < self.start_timesteps:
            return cur + self.start_timesteps
        return cur

    def _serializable_hparams(self) -> dict[str, Any]:
        hp_dict = asdict(self.hp)
        return {k: v for k, v in hp_dict.items() if not k.startswith('_')}

    def _save_with_metadata(self, model_path: str, eval_reward: float,
                            batch_idx: int, episode_idx: int):
        self.model.save(model_path)

        metadata = {
            'current_reward': float(eval_reward),
            'best_reward': float(self.best_eval_reward),
            'episode': int(episode_idx),
            'best_episode': int(self.best_eval_episode),
            'batch_idx': int(batch_idx),
            'timesteps': int(self._total_timesteps()),
            'hyperparameters': self._serializable_hparams(),
            'config_content': self.config_content,
            'config_path': self.cfg_path,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        _save_metadata(model_path, metadata)

    def _run_eval_and_checkpoint(self):
        mean_reward, _ = sb3_evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=1,
            deterministic=True,
            render=False,
            return_episode_rewards=False
        )

        self.latest_eval_reward = float(mean_reward)

        total_timesteps = self._total_timesteps()
        episodes_done = int(total_timesteps / max(1, self.hp.actions_per_episode))
        batch_idx = int(total_timesteps / max(1, self.hp.rollout_size))

        # Keep training output concise (TorchRL-like): do not print a
        # per-evaluation summary line here.

        model_batch_path = os.path.join(self.checkpoint_dir,
                                        f'model-{batch_idx}.zip')
        self._save_with_metadata(
            model_batch_path,
            self.latest_eval_reward,
            batch_idx,
            episodes_done
        )
        self._save_with_metadata(
            self.latest_model_path,
            self.latest_eval_reward,
            batch_idx,
            episodes_done
        )

        if self.latest_eval_reward > self.best_eval_reward:
            self.best_eval_reward = self.latest_eval_reward
            self.best_eval_episode = episodes_done
            if self.verbose:
                print(
                    f'New best eval reward: {self.best_eval_reward:.6f} '
                    f'at episode {episodes_done}'
                )
            self._save_with_metadata(
                self.best_model_path,
                self.latest_eval_reward,
                batch_idx,
                episodes_done
            )

        self._latest_eval_str = (
            f'eval reward: {self.latest_eval_reward:.5f} '
            f'(best: {self.best_eval_reward:.5f})'
        )
        self._set_progress_postfix()

    def _on_training_start(self):
        initial = min(max(0, self._episodes_shown), self.hp.episodes)
        self._episodes_shown = initial
        self._pbar = tqdm(total=self.hp.episodes, desc='Training', initial=initial)
        self._set_progress_postfix()
        return None

    def _on_step(self):
        if self.num_timesteps - self._last_eval_timestep >= self.eval_freq_steps:
            self._run_eval_and_checkpoint()
            self._last_eval_timestep = int(self.num_timesteps)

        return True

    def _on_rollout_end(self):
        rb = getattr(self.model, 'rollout_buffer', None)
        if rb is not None and hasattr(rb, 'rewards'):
            try:
                self._latest_train_reward = float(rb.rewards.mean())
            except Exception:
                pass

        self._sync_progress_bar()
        self._set_progress_postfix()
        return None

    def _on_training_end(self):
        # Ensure at least one checkpoint exists even if eval frequency is large.
        if self.latest_eval_reward is None:
            self._run_eval_and_checkpoint()

        self._sync_progress_bar()
        if self._pbar is not None:
            self._pbar.close()


def train_agent(mesh_file, cfg_file, backend_name,
                checkpoint_dir='checkpoints', ic_dir=None, load_model=None):
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

    num_devices = max(1, get_device_count(backend_name))
    num_envs = max(1, num_devices * max(1, hp.envs_per_device))

    hp._calculate_derived(probe_env, num_envs)
    hp.print_summary(num_devices=num_devices, num_envs=num_envs)

    probe_env.close()

    print(f"\nFound {num_devices} devices for backend '{backend_name}'")
    print(f'Using SB3 with {num_envs} environment(s).')

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

    start_timesteps = 0
    start_episode = 0
    best_reward = float('-inf')

    model = None
    resolved_model_path = None

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

        print(f'Loading SB3 model: {resolved_model_path}')
        model = PPO.load(
            resolved_model_path,
            env=train_env,
            device=hp.torch_device,
            print_system_info=False
        )
        model.verbose = 0

    if model is None:
        model = PPO(
            policy='MlpPolicy',
            env=train_env,
            learning_rate=hp.lr,
            n_steps=hp.n_steps,
            batch_size=hp.batch_size,
            n_epochs=hp.num_epochs,
            gamma=hp.gamma,
            gae_lambda=hp.lmbda,
            clip_range=hp.clip_epsilon,
            ent_coef=hp.entropy_eps,
            max_grad_norm=hp.max_grad_norm,
            use_sde=hp.use_sde,
            sde_sample_freq=hp.sde_sample_freq,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log_root,
            device=hp.torch_device,
            seed=hp.seed,
            verbose=0,
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
