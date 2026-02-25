from dataclasses import dataclass

from pyfr.inifile import Inifile

from .utils import get_closest_divisor


@dataclass
class HyperParameters:
    # General settings
    algorithm: str = 'ppo'
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

    # PPO-LSTM policy parameters (sb3-contrib RecurrentPPO)
    lstm_hidden_size: int = 256
    n_lstm_layers: int = 1
    shared_lstm: bool = False
    enable_critic_lstm: bool = True
    lstm_dropout: float = 0.0

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
        algo_key = (self.algorithm or '').strip().lower().replace('-', '_')
        is_recurrent = algo_key == 'ppo_lstm'

        section_items = [
            ('General Settings', [
                ('algorithm', "RL algorithm ('ppo' or 'ppo-lstm')"),
                ('torch_device', "'cuda', 'cpu', or 'auto'"),
                ('print_config_on_load', 'Print config file content on model load')
            ]),
            ('Runtime Parallelism', [
                ('num_devices', 'Detected backend devices'),
                ('envs_per_device', 'Environments spawned per detected device'),
                ('num_envs', 'Total environments in vectorized collector'),
                ('seed', 'SB3 random seed')
            ]),
            ('Network Architecture', [
                ('num_hidden_layers_policy', 'No. of hidden layers in policy network'),
                ('num_hidden_layers_value', 'No. of hidden layers in value network'),
                ('num_cells_policy', 'Size of policy network hidden layers'),
                ('num_cells_value', 'Size of value network hidden layers'),
                ('activation_policy', 'Activation function used by SB3 policy/value'),
                ('activation_value', 'Read from config; SB3 shares activation_fn')
            ]),
            ('Training Schedule', [
                ('episodes', 'Total training episodes target'),
                ('episodes_per_batch', 'Episodes per policy update target'),
                ('desired_num_minibatches', 'Target minibatches per update'),
                ('num_epochs', 'Training epochs per update')
            ]),
            ('PPO Parameters', [
                ('clip_epsilon', 'PPO clipping parameter'),
                ('gamma', 'Discount factor'),
                ('lmbda', 'GAE lambda parameter'),
                ('entropy_eps', 'Entropy bonus coefficient'),
                ('lr', 'Learning rate'),
                ('max_grad_norm', 'Gradient clipping norm'),
                ('use_sde', 'Enable generalized State-Dependent Exploration'),
                ('sde_sample_freq', 'Noise resample frequency (-1 = per rollout)'),
                ('squash_output', 'Tanh-squash actions (requires use_sde=true)')
            ])
        ]

        if is_recurrent:
            section_items.append((
                'PPO-LSTM Parameters',
                [
                    ('lstm_hidden_size', 'LSTM hidden size (RecurrentPPO only)'),
                    ('n_lstm_layers', 'Number of LSTM layers (RecurrentPPO only)'),
                    ('shared_lstm', 'Share LSTM between actor and critic'),
                    ('enable_critic_lstm', 'Use separate critic LSTM'),
                    ('lstm_dropout', 'Dropout inside LSTM stack')
                ]
            ))

        section_items.extend([
            ('SB3 Derived Values', [
                ('actions_per_episode', 'Actions per episode'),
                ('frames_per_batch', 'Target frames per update'),
                ('total_frames', 'Target total training frames'),
                ('n_steps', 'SB3 rollout steps per environment'),
                ('rollout_size', 'SB3 rollout size = n_steps * num_envs'),
                ('batch_size', 'SB3 minibatch size')
            ]),
            ('Evaluation Settings', [
                ('eval_frequency', 'Evaluate every N update cycles')
            ])
        ])

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

        for section_name, params in section_items:
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

        if not is_recurrent:
            recurrent_keys = (
                'lstm_hidden_size', 'n_lstm_layers', 'shared_lstm',
                'enable_critic_lstm', 'lstm_dropout'
            )
            if any(self._param_sources.get(k) == 'config' for k in recurrent_keys):
                print(
                    '\nNote: PPO-LSTM-specific parameters are set but '
                    "algorithm != 'ppo-lstm'; they will be ignored."
                )
