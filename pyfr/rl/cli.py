#!/usr/bin/env python
from argparse import ArgumentParser, FileType
import os

import mpi4py.rc
mpi4py.rc.initialize = False
from pyfr.mpiutil import get_comm_rank_root, init_mpi

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
    ap_train.set_defaults(process=process_train)

    # Add backend argument
    backends = sorted(cls.name for cls in subclasses(BaseBackend))
    ap_train.add_argument('--backend', '-b', choices=backends, required=True,
                         help='backend to use')
    
    # Evaluate command
    ap_eval = sp.add_parser('evaluate', help='evaluate trained policy')
    ap_eval.add_argument('mesh', help='mesh file')
    ap_eval.add_argument('cfg', type=FileType('r'), help='config file')
    ap_eval.add_argument('--model-path', required=True, help='path to model checkpoint')
    ap_eval.add_argument('--episodes', type=int, default=10, help='number of evaluation episodes')
    ap_eval.add_argument('--backend', '-b', choices=backends, required=True)
    ap_eval.set_defaults(process=process_evaluate)


    # Parse args
    args = ap.parse_args()

    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()

def process_train(args):
    # Manually initialise MPI
    init_mpi()

    print(f"Starting training with checkpoint dir: {args.checkpoint_dir}")
    from .train import train_agent

    train_agent(
        mesh_file=args.mesh,
        cfg_file=args.cfg,
        backend_name=args.backend,
        checkpoint_dir=args.checkpoint_dir
    )

def process_evaluate(args):
    init_mpi()
    
    from .train import evaluate_policy
    from .env import PyFREnvironment
    
    env = PyFREnvironment(args.mesh, args.cfg, args.backend)
    evaluate_policy(env, args.model_path, args.episodes)
    env.close()

if __name__ == '__main__':
    main()