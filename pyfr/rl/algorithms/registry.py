from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    display_name: str
    policy_name: str
    recurrent: bool


_ALIAS_TO_CANONICAL = {
    'ppo': 'ppo',
    'ppo_lstm': 'ppo_lstm',
    'ppo-lstm': 'ppo_lstm',
    'ppolstm': 'ppo_lstm',
    'recurrentppo': 'ppo_lstm',
    'recurrent-ppo': 'ppo_lstm',
}

_SPECS = {
    'ppo': AlgorithmSpec(
        name='ppo',
        display_name='PPO',
        policy_name='MlpPolicy',
        recurrent=False,
    ),
    'ppo_lstm': AlgorithmSpec(
        name='ppo_lstm',
        display_name='PPO-LSTM',
        policy_name='MlpLstmPolicy',
        recurrent=True,
    ),
}


def normalize_algorithm_name(name: str | None) -> str:
    key = (name or 'ppo').strip().lower().replace(' ', '').replace('/', '-')
    canonical = _ALIAS_TO_CANONICAL.get(key)
    if canonical is None:
        valid = ', '.join(sorted(set(_ALIAS_TO_CANONICAL.keys())))
        raise ValueError(f"Unknown algorithm '{name}'. Valid options: {valid}")
    return canonical


def get_algorithm_spec(name: str | None) -> AlgorithmSpec:
    canonical = normalize_algorithm_name(name)
    return _SPECS[canonical]


def is_recurrent_algorithm(name: str | None) -> bool:
    return get_algorithm_spec(name).recurrent


def _model_class(name: str):
    canonical = normalize_algorithm_name(name)
    if canonical == 'ppo':
        from stable_baselines3 import PPO
        return PPO
    if canonical == 'ppo_lstm':
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO

    raise ValueError(f'Unsupported algorithm: {name}')


def create_model(name: str, env, hp, policy_kwargs, tensorboard_log: str):
    spec = get_algorithm_spec(name)
    model_cls = _model_class(spec.name)
    return model_cls(
        policy=spec.policy_name,
        env=env,
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
        tensorboard_log=tensorboard_log,
        device=hp.torch_device,
        seed=hp.seed,
        verbose=0,
    )


def load_model(name: str, model_path: str, env, device: str):
    model_cls = _model_class(name)
    return model_cls.load(
        model_path,
        env=env,
        device=device,
        print_system_info=False
    )

