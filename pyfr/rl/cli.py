#!/usr/bin/env python
from argparse import ArgumentParser, FileType
import os

import mpi4py.rc
mpi4py.rc.initialize = False
from pyfr.mpiutil import get_comm_rank_root, init_mpi

from pyfr._version import __version__
from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.plugins import BaseCLIPlugin
from pyfr.progress import ProgressBar, ProgressSequenceAction
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import (BaseWriter, get_writer_by_extn, get_writer_by_name,
                          write_pyfrms)

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
    ap_train.add_argument('--restart', help='restart solution file')
    ap_train.add_argument('--load-model', help='load existing model checkpoint to continue training')
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
    ap_eval.add_argument('--restart', help='restart solution file')
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

    # Load restart solution if provided
    restart_soln = None
    if args.restart:
        restart_soln = NativeReader(args.restart)
        mesh = NativeReader(args.mesh)
        
        # Verify mesh UUID matches
        if restart_soln['mesh_uuid'] != mesh['mesh_uuid']:
            raise RuntimeError('Restart solution does not match mesh.')

    print(f"Starting training with checkpoint dir: {args.checkpoint_dir}")
    from .train import train_agent

    train_agent(
        mesh_file=args.mesh,
        cfg_file=args.cfg,
        backend_name=args.backend,
        checkpoint_dir=args.checkpoint_dir,
        restart_soln=restart_soln,
        load_model=args.load_model
    )

def process_evaluate(args):
    init_mpi()
    
    # Load restart solution if provided 
    restart_soln = None
    if args.restart:
        restart_soln = NativeReader(args.restart)
        mesh = NativeReader(args.mesh)
        
        if restart_soln['mesh_uuid'] != mesh['mesh_uuid']:
            raise RuntimeError('Restart solution does not match mesh.')
            
    from .evaluate import evaluate_policy
    from .env import PyFREnvironment
    
    env = PyFREnvironment(
        args.mesh,
        args.cfg,
        args.backend,
        restart_soln=restart_soln
    )
    evaluate_policy(env, args.model_path, args.episodes)
    env.close()