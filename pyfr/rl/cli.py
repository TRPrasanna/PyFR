#!/usr/bin/env python
from argparse import ArgumentParser, FileType

import mpi4py.rc
mpi4py.rc.initialize = False

from pyfr._version import __version__
from pyfr.backends import BaseBackend
from pyfr.util import subclasses

def main():
    ap = ArgumentParser(prog='pyfr-rl')
    sp = ap.add_subparsers(help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')
    ap.add_argument('--version', '-V', action='version',
                   version=f'%(prog)s {__version__}')
    ap.add_argument('--progress', '-p', action='store_true',
                   help='show progress')

    # Train command
    ap_train = sp.add_parser('train', help='train DRL policy')
    ap_train.add_argument('mesh', help='mesh file')
    ap_train.add_argument('cfg', type=FileType('r'), help='config file')
    ap_train.add_argument('--checkpoint-dir', default='checkpoints',
                         help='directory to save checkpoints')
    ap_train.add_argument('--ic-dir', default=None,
                         help='directory of initial condition snapshots')
    ap_train.add_argument('--load-model', help='load existing model checkpoint to continue training')
    ap_train.add_argument(
        '--algorithm',
        default=None,
        help='RL algorithm override (e.g., ppo, ppo-lstm)'
    )
    ap_train.set_defaults(process=process_train)

    # Add backend argument
    backends = sorted(cls.name for cls in subclasses(BaseBackend))
    ap_train.add_argument('--backend', '-b', choices=backends, required=True,
                         help='backend to use')
    
    # Evaluate command
    ap_eval = sp.add_parser('evaluate', help='evaluate trained policy')
    ap_eval.add_argument('mesh', help='mesh file')
    ap_eval.add_argument('cfg', type=FileType('r'), help='config file')
    ap_eval.add_argument('--episodes', type=int, default=1,
                        help='number of evaluation episodes (default: 1)')
    ap_eval.add_argument(
        '--load-model',
        default=None,
        help='path to model checkpoint; if omitted, evaluate a fresh untrained policy'
    )
    ap_eval.add_argument('--ic-dir', default=None,
                        help='directory of initial condition snapshots (optional)')
    ap_eval.add_argument(
        '--algorithm',
        default=None,
        help='RL algorithm override (e.g., ppo, ppo-lstm)'
    )
    ap_eval.add_argument(
        '--stochastic',
        action='store_true',
        help='sample actions stochastically instead of using deterministic evaluation'
    )
    ap_eval.add_argument('--backend', '-b', choices=backends, required=True)
    ap_eval.set_defaults(process=process_evaluate)

    # HPO command
    ap_hpo = sp.add_parser('hpo', help='run hyperparameter optimization (Optuna)')
    ap_hpo.add_argument('mesh', help='mesh file')
    ap_hpo.add_argument('cfg', type=FileType('r'), help='config file')
    ap_hpo.add_argument('--checkpoint-dir', default='hpo-runs',
                        help='directory to save HPO artifacts')
    ap_hpo.add_argument('--ic-dir', default=None,
                        help='directory of initial condition snapshots')
    ap_hpo.add_argument(
        '--algorithm',
        default=None,
        help='RL algorithm override (e.g., ppo, ppo-lstm)'
    )
    ap_hpo.add_argument('--study-name', default=None,
                        help='Optuna study name')
    ap_hpo.add_argument('--storage', default=None,
                        help='Optuna storage URL (e.g., sqlite:///hpo.db)')
    ap_hpo.add_argument('--n-trials', type=int, default=None,
                        help='number of HPO trials')
    ap_hpo.add_argument('--timeout', type=int, default=None,
                        help='time limit in seconds for HPO')
    ap_hpo.add_argument('--sampler', choices=['tpe', 'random'], default=None,
                        help='Optuna sampler')
    ap_hpo.add_argument('--pruner', choices=['hyperband', 'none'], default=None,
                        help='Optuna pruner')
    ap_hpo.add_argument('--device-id', type=int, default=None,
                        help='starting backend device id for this worker')
    ap_hpo.add_argument('--envs-per-trial', type=int, default=None,
                        help='vectorized environments per trial')
    ap_hpo.add_argument('--episodes-per-batch', type=int, default=None,
                        help='episodes collected per update during HPO')
    ap_hpo.add_argument('--trial-updates', type=int, default=None,
                        help='number of PPO updates per trial')
    ap_hpo.add_argument('--backend', '-b', choices=backends, required=True)
    ap_hpo.set_defaults(process=process_hpo)


    # Parse args
    args = ap.parse_args()

    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()

def process_train(args):
    from pyfr.mpiutil import get_comm_rank_root, init_mpi
    init_mpi()
    _, rank, root = get_comm_rank_root()
    if rank == root:
        print(f"Starting training with checkpoint dir: {args.checkpoint_dir}")
    from .train import train_agent

    train_agent(
        mesh_file=args.mesh,
        cfg_file=args.cfg,
        backend_name=args.backend,
        checkpoint_dir=args.checkpoint_dir,
        ic_dir=args.ic_dir,
        load_model=args.load_model,
        algorithm=args.algorithm
    )

def process_evaluate(args):
    from .evaluate import evaluate_policy
    evaluate_policy(
        mesh_file=args.mesh,
        cfg_file=args.cfg,
        backend_name=args.backend,
        load_model=args.load_model,
        ic_dir=args.ic_dir,
        episodes=args.episodes,
        algorithm=args.algorithm,
        stochastic=args.stochastic
    )


def process_hpo(args):
    from .hpo import run_hpo

    run_hpo(
        mesh_file=args.mesh,
        cfg_file=args.cfg,
        backend_name=args.backend,
        checkpoint_dir=args.checkpoint_dir,
        ic_dir=args.ic_dir,
        algorithm=args.algorithm,
        study_name=args.study_name,
        storage=args.storage,
        n_trials=args.n_trials,
        timeout=args.timeout,
        sampler=args.sampler,
        pruner=args.pruner,
        device_id=args.device_id,
        envs_per_trial=args.envs_per_trial,
        episodes_per_batch=args.episodes_per_batch,
        trial_updates=args.trial_updates
    )
