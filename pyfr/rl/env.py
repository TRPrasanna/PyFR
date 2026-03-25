import gc
import os
import random
import sys
import traceback
from datetime import datetime
from typing import List

import h5py
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs.common import EnvBase

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver


_CMD_RESET = 'reset'
_CMD_STEP = 'step'
_CMD_SET_MODE = 'set-mode'
_CMD_CLOSE = 'close'
_CMD_STOP = 'stop'


class PyFREnvironment(EnvBase):
    """PyFR environment compatible with TorchRL."""

    def __init__(self, mesh_file, cfg_file, backend_name, device_id,
                 ic_dir=None, print_diagnostic=False):
        init_mpi()
        self.comm, self.rank, self.root = get_comm_rank_root()
        self.is_root = self.rank == self.root
        self.print_diagnostic = print_diagnostic and self.is_root

        device = torch.device('cpu')
        super().__init__(device=device)

        # Keep one mesh reader open and reuse it for loading restart solutions
        self.mesh_reader = NativeReader(mesh_file)
        self.mesh = self.mesh_reader.mesh
        self.cfg = Inifile.load(cfg_file)

        if backend_name in {'hip', 'cuda'} and device_id is not None:
            self.cfg.set(f'backend-{backend_name}', 'device-id', device_id)
            if self.print_diagnostic:
                print(f'Using {backend_name} device {device_id}')

        self.backend = get_backend(backend_name, self.cfg)

        self.tend = self.cfg.getfloat('solver-time-integrator', 'tend')
        tstart = self.cfg.getfloat('solver-time-integrator', 'tstart', 0.0)

        self.num_control_actions = self.cfg.getint(
            'solver-plugin-reinforcementlearning', 'num-control-actions'
        )
        self.actions_low = self.cfg.getliteral(
            'solver-plugin-reinforcementlearning', 'actions-low'
        )
        self.actions_high = self.cfg.getliteral(
            'solver-plugin-reinforcementlearning', 'actions-high'
        )
        self.actions_init = self.cfg.getliteral(
            'solver-plugin-reinforcementlearning', 'actions-init'
        )

        if not (
            len(self.actions_low)
            == len(self.actions_high)
            == len(self.actions_init)
            == self.num_control_actions
        ):
            raise ValueError(
                'Action bounds/init sizes do not match num-control-actions'
            )

        if self.print_diagnostic:
            print(f'Number of control actions: {self.num_control_actions}')
            for i in range(self.num_control_actions):
                print(
                    f'Control action {i + 1} range: '
                    f'{self.actions_low[i]} to {self.actions_high[i]}'
                )

        self.current_control = np.array(self.actions_init, dtype=np.float64)
        self.previous_control = np.array(self.actions_init, dtype=np.float64)

        # dtend is custom in this RL workflow; fall back to (tend - tstart)
        self.dtend = self.cfg.getfloat(
            'solver-time-integrator', 'dtend', self.tend - tstart
        )
        self.eval_time = self.cfg.getfloat(
            'solver-plugin-reinforcementlearning', 'eval-time', self.dtend
        )
        self._current_time_limit = self.dtend

        self.action_interval = self.cfg.getfloat(
            'solver-plugin-reinforcementlearning', 'action-interval'
        )
        self.max_training_steps = int(self.dtend / self.action_interval)
        self.max_eval_steps = int(self.eval_time / self.action_interval)
        self._current_max_steps = self.max_training_steps

        self.step_count = -1

        self.ic_manager = None
        if self.is_root and ic_dir is not None:
            try:
                self.ic_manager = InitialConditionManager(
                    ic_dir, self.mesh.uuid,
                    print_diagnostic=self.print_diagnostic
                )
            except ValueError as e:
                print(f'\nWarning: {e}')
                print('Continuing without initial condition snapshots...')
        elif self.print_diagnostic:
            print('\nNote: No initial condition directory provided.')
            print('Training will use default initial conditions.')

        self.is_evaluating = False

        restart_soln = self._load_restart_soln()
        self._init_solver(initsoln=restart_soln)

        obs_size = self.rl_plugin.observation_size
        if self.print_diagnostic:
            try:
                var_list = self.rl_plugin.obs_var_names
                print(f"Observation variables: {', '.join(var_list)}")
            except AttributeError:
                print(
                    'Observation variables: '
                    '<plugin does not expose obs_var_names>'
                )

            print(f'Observation size: {obs_size}')
            print(f'Reward function: {self.rl_plugin.reward_function}')
            print(
                'Variables used in reward function: '
                f"{', '.join(sorted(self.rl_plugin.used_variables))}"
            )

        self.observation_spec = Composite(
            {
                'observation': Unbounded(
                    shape=(obs_size,), device=self.device
                )
            },
            shape=torch.Size([])
        )

        self.state_spec = self.observation_spec.clone()

        self.action_spec = Composite(
            {
                'action': Bounded(
                    low=torch.tensor(self.actions_low, device=self.device),
                    high=torch.tensor(self.actions_high, device=self.device),
                    shape=(self.num_control_actions,),
                    device=self.device
                )
            },
            batch_size=torch.Size([])
        )

        self.reward_spec = Composite(
            {
                'reward': Unbounded(shape=(1,), device=self.device)
            },
            shape=torch.Size([])
        )

        self.full_done_spec = Composite(
            {
                'done': Categorical(
                    n=2, shape=(1,),
                    dtype=torch.bool, device=self.device
                ),
                'terminated': Categorical(
                    n=2, shape=(1,),
                    dtype=torch.bool, device=self.device
                ),
                'truncated': Categorical(
                    n=2, shape=(1,),
                    dtype=torch.bool, device=self.device
                ),
            },
            shape=torch.Size([])
        )

        if self.print_diagnostic:
            print('Environment initialized.')

        self.episode_count = 0
        self.pbar = None
        self.count_episodes = True

    def set_progress_bar(self, pbar):
        self.pbar = pbar

    def _load_restart_soln(self):
        payload = {'ic_file': None, 'error': None}

        if self.is_root and self.ic_manager is not None:
            try:
                if self.is_evaluating:
                    payload['ic_file'] = self.ic_manager.get_eval_ic()
                else:
                    payload['ic_file'] = self.ic_manager.get_random_ic()
            except Exception as e:
                payload['error'] = str(e)

        payload = self.comm.bcast(payload, root=self.root)

        if payload['error'] is not None:
            if self.is_root:
                print(f"Warning: Failed to load IC file: {payload['error']}")
                print('Using default initial conditions.')
            return None

        ic_file = payload['ic_file']
        if ic_file is None:
            return None

        return self.mesh_reader.load_soln(ic_file)

    def _init_solver(self, initsoln=None):
        self._release_solver()
        self.restart_soln = initsoln

        self.solver = get_solver(
            self.backend, self.mesh, self.restart_soln, self.cfg
        )

        # The RL BC hooks and plugin read controls from env.
        self.solver.env = self
        self.solver.system.env = self

        self.rl_plugin = next(
            p for p in self.solver.plugins if p.name == 'reinforcementlearning'
        )

        self.current_time = self.solver.tcurr
        self.max_time = self.current_time + self._current_time_limit

    def _release_solver(self):
        if not hasattr(self, 'solver') or self.solver is None:
            return

        try:
            self.backend.wait()
        except Exception:
            pass

        self.rl_plugin = None
        old_solver = self.solver
        self.solver = None
        del old_solver
        gc.collect()

    def _get_observation_size(self):
        return self.rl_plugin.observation_size

    def _reset(self, tensordict=None, **kwargs):
        self.step_count = 0
        self.current_control = np.array(self.actions_init, dtype=np.float64)
        self.previous_control = np.array(self.actions_init, dtype=np.float64)

        restart_soln = self._load_restart_soln()
        self._init_solver(initsoln=restart_soln)

        self.rl_plugin.reset()
        observation = self._get_observation()

        return TensorDict(
            {
                'observation': observation,
                'done': torch.tensor(False, device=self.device,
                                     dtype=torch.bool),
                'terminated': torch.tensor(False, device=self.device,
                                           dtype=torch.bool),
                'truncated': torch.tensor(False, device=self.device,
                                          dtype=torch.bool),
            },
            batch_size=torch.Size([])
        )

    def _step(self, tensordict):
        self.previous_control = self.current_control
        self.current_control = tensordict['action'].detach().cpu().numpy()

        if 'step_count' in tensordict.keys(True):
            self.step_count = int(tensordict['step_count'].item())
        else:
            self.step_count += 1

        if np.isnan(self.current_control).any():
            raise RuntimeError('Control signal is NaN. Aborting.')

        self.current_time = self.solver.tcurr
        self.next_action_time = self.current_time + self.action_interval

        try:
            self.solver.advance_to(self.next_action_time)

            reward = self._compute_reward()
            observation = self._get_observation()
            truncated = self._check_done()
            terminated = False

            if truncated and self.count_episodes and self.is_root:
                self.episode_count += 1

        except RuntimeError as e:
            if self.is_root:
                print(
                    f'Solver crashed: {e}. '
                    f'Last actions were: {self.current_control}'
                )

            observation = self.observation_spec.zero(
                torch.Size([])
            )['observation']
            reward = -10.0
            truncated = False
            terminated = True

            if self.count_episodes and self.is_root:
                self.episode_count += 1

        return TensorDict(
            {
                'observation': observation,
                'reward': torch.tensor([reward], device=self.device),
                'done': torch.tensor([terminated or truncated],
                                     device=self.device, dtype=torch.bool),
                'terminated': torch.tensor([terminated],
                                           device=self.device,
                                           dtype=torch.bool),
                'truncated': torch.tensor([truncated],
                                          device=self.device,
                                          dtype=torch.bool),
            },
            batch_size=tensordict.shape
        )

    def _get_observation(self):
        obs = self.rl_plugin._get_observation(self.solver)

        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)

        return torch.tensor(obs, device=self.device).float()

    def _compute_reward(self):
        return float(self.rl_plugin._get_reward(self.solver))

    def _check_done(self) -> bool:
        return self.step_count + 1 >= self._current_max_steps

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def set_evaluation_mode(self, is_evaluating: bool):
        self.is_evaluating = is_evaluating
        self._current_time_limit = (
            self.eval_time if is_evaluating else self.dtend
        )
        self._current_max_steps = (
            self.max_eval_steps if is_evaluating else self.max_training_steps
        )
        self.count_episodes = not is_evaluating

    def close(self, *, raise_if_closed=True, **kwargs):
        self._release_solver()
        try:
            self.mesh_reader.close()
        except Exception:
            pass
        super().close(raise_if_closed=raise_if_closed)


class CollectiveEnvController(EnvBase):
    """Root-rank controller that broadcasts env commands to MPI workers."""

    def __init__(self, env_name: str, env: PyFREnvironment):
        super().__init__(device=env.device)

        self.env_name = env_name
        self.env = env
        self.comm = env.comm
        self.rank = env.rank
        self.root = env.root

        if self.rank != self.root:
            raise RuntimeError(
                'CollectiveEnvController can only be created on the root rank.'
            )

        self.observation_spec = env.observation_spec.clone()
        self.state_spec = env.state_spec.clone()
        self.action_spec = env.action_spec.clone()
        self.reward_spec = env.reward_spec.clone()
        self.full_done_spec = env.full_done_spec.clone()
        self._closed = False

    def _broadcast(self, payload):
        self.comm.bcast(payload, root=self.root)

    def _reset(self, tensordict=None, **kwargs):
        self._broadcast({
            'cmd': _CMD_RESET,
            'env': self.env_name,
        })
        return self.env._reset(tensordict, **kwargs)

    def _step(self, tensordict):
        action = tensordict['action'].detach().cpu().numpy()
        self._broadcast({
            'cmd': _CMD_STEP,
            'env': self.env_name,
            'action': action,
        })
        return self.env._step(tensordict)

    def _set_seed(self, seed):
        self.env._set_seed(seed)

    def set_evaluation_mode(self, is_evaluating: bool):
        self._broadcast({
            'cmd': _CMD_SET_MODE,
            'env': self.env_name,
            'is_evaluating': bool(is_evaluating),
        })
        self.env.set_evaluation_mode(is_evaluating)

    def close(self, *, raise_if_closed=True, **kwargs):
        if self._closed:
            return

        self._broadcast({
            'cmd': _CMD_CLOSE,
            'env': self.env_name,
        })
        self.env.close(raise_if_closed=False)
        self._closed = True
        super().close(raise_if_closed=False)


def stop_collective_workers(comm, root: int):
    comm.bcast({'cmd': _CMD_STOP}, root=root)


def serve_collective_envs(envs: dict[str, PyFREnvironment]):
    if not envs:
        return

    sample_env = next(iter(envs.values()))
    comm = sample_env.comm
    root = sample_env.root
    rank = sample_env.rank

    if rank == root:
        raise RuntimeError('serve_collective_envs must run on non-root ranks.')

    active_envs = dict(envs)

    try:
        while True:
            payload = comm.bcast(None, root=root)
            cmd = payload.get('cmd')

            if cmd == _CMD_STOP:
                break

            env_name = payload.get('env')
            if env_name not in active_envs:
                raise RuntimeError(
                    f"Unknown collective environment '{env_name}'."
                )

            env = active_envs[env_name]

            if cmd == _CMD_RESET:
                env._reset()
            elif cmd == _CMD_STEP:
                action = payload['action']
                td = TensorDict(
                    {'action': torch.tensor(action, device=env.device)},
                    batch_size=torch.Size([])
                )
                env._step(td)
            elif cmd == _CMD_SET_MODE:
                env.set_evaluation_mode(payload['is_evaluating'])
            elif cmd == _CMD_CLOSE:
                env.close(raise_if_closed=False)
                active_envs.pop(env_name, None)
            else:
                raise RuntimeError(f"Unknown collective command '{cmd}'.")
    except BaseException as exc:
        exc_name = type(exc).__name__
        cmd = (
            payload.get('cmd')
            if 'payload' in locals() and payload else '<none>'
        )
        env_name = (
            payload.get('env')
            if 'payload' in locals() and payload else '<none>'
        )
        print(
            '[mpi-worker] '
            f'rank={rank} env={env_name} cmd={cmd} '
            f'error={exc_name}: {exc}',
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        sys.stderr.flush()

        try:
            comm.Abort(1)
        finally:
            raise
    finally:
        for env in active_envs.values():
            try:
                env.close(raise_if_closed=False)
            except Exception:
                pass


class InitialConditionManager:
    def __init__(self, ic_dir: str, mesh_uuid: str, print_diagnostic=False):
        self.ic_dir = ic_dir
        self.mesh_uuid = mesh_uuid
        self.print_diagnostic = print_diagnostic
        self.ic_files = self._find_valid_ics()
        self.unused_files = set(self.ic_files)
        self.eval_ic_file = self._get_oldest_ic()

        if self.print_diagnostic:
            print(
                f'\nFound {len(self.ic_files)} initial condition files '
                f'in {ic_dir}'
            )
        if self.eval_ic_file and self.print_diagnostic:
            print(
                f'Using {os.path.basename(self.eval_ic_file)} for evaluation'
            )

    def _get_oldest_ic(self) -> str:
        if not self.ic_files:
            return None

        oldest = min(self.ic_files, key=os.path.getctime)
        if self.print_diagnostic:
            creation_time = datetime.fromtimestamp(os.path.getctime(oldest))
            print(f'\nSelected evaluation IC: {os.path.basename(oldest)}')
            print(f'Creation time: {creation_time:%Y-%m-%d %H:%M:%S}')

        return oldest

    def get_eval_ic(self) -> str:
        return self.eval_ic_file

    def _find_valid_ics(self) -> List[str]:
        if not os.path.exists(self.ic_dir):
            raise ValueError(f'IC directory {self.ic_dir} not found')

        ic_files = []
        for f in sorted(os.listdir(self.ic_dir)):
            if not f.endswith('.pyfrs'):
                continue

            file_path = os.path.join(self.ic_dir, f)

            try:
                with h5py.File(file_path, 'r') as soln:
                    muuid = soln['mesh-uuid'][()].decode()

                if muuid == self.mesh_uuid:
                    ic_files.append(file_path)
            except Exception:
                continue

        return ic_files

    def get_random_ic(self) -> str:
        if not self.unused_files:
            self.unused_files = set(self.ic_files)

        if not self.unused_files:
            return None

        ic_file = random.choice(sorted(self.unused_files))
        self.unused_files.remove(ic_file)
        return ic_file
