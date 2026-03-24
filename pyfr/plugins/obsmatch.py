import csv
from pathlib import Path

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BaseSolnPlugin, PostactionMixin, init_csv
from pyfr.plugins.sampler import _process_con_to_pri
from pyfr.points import PointSampler
from pyfr.util import first
from pyfr.writers.native import NativeWriter


def _load_csv_target(path, obs_vars):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f'Empty target CSV file: {path}')

    coord_names = [c for c in ('x', 'y', 'z') if c in rows[0]]
    if len(coord_names) not in {2, 3}:
        raise ValueError('Target CSV must contain x,y or x,y,z columns')

    if obs_vars is None:
        obs_vars = [k for k in rows[0] if k not in coord_names]
        if not obs_vars:
            raise ValueError('Target CSV does not contain any observation columns')

    pts = np.array([[float(r[c]) for c in coord_names] for r in rows], dtype=float)
    vals = np.array([[float(r[v]) for v in obs_vars] for r in rows], dtype=float)

    return pts, vals, list(obs_vars)


def _load_npz_target(path, obs_vars, points_key, obs_key, values_key, vars_key):
    with np.load(path, allow_pickle=False) as data:
        if points_key not in data:
            raise ValueError(f"Target NPZ is missing '{points_key}'")

        pts = np.asarray(data[points_key], dtype=float)
        file_vars = None
        if vars_key in data:
            file_vars = [str(v) for v in np.asarray(data[vars_key]).tolist()]

        if values_key in data:
            vals = np.asarray(data[values_key], dtype=float)
            if vals.ndim != 2:
                raise ValueError(f"Target NPZ '{values_key}' must be 2D")
        elif obs_key in data:
            obs = np.asarray(data[obs_key], dtype=float).reshape(-1)
            if file_vars is None:
                if obs_vars is None:
                    raise ValueError(
                        'Target NPZ with flattened obs requires variables in file '
                        'or observation-variables in config'
                    )
                nfilevars = len(obs_vars)
            else:
                nfilevars = len(file_vars)

            if len(obs) != len(pts) * nfilevars:
                raise ValueError('Flattened target obs has inconsistent length')

            vals = obs.reshape(len(pts), nfilevars)
        else:
            raise ValueError(
                f"Target NPZ must contain either '{values_key}' or '{obs_key}'"
            )

    if obs_vars is None:
        if file_vars is None:
            raise ValueError(
                'Target NPZ requires variables in file or observation-variables '
                'in config'
            )
        obs_vars = list(file_vars)

    if file_vars is None:
        if vals.shape[1] != len(obs_vars):
            raise ValueError('Target values columns do not match observation vars')
        return pts, vals, list(obs_vars)

    try:
        vidxs = [file_vars.index(v) for v in obs_vars]
    except ValueError as err:
        raise ValueError(
            f'Target NPZ variables {file_vars} do not contain requested '
            f'observation vars {list(obs_vars)}'
        ) from err

    return pts, vals[:, vidxs], list(obs_vars)


class ObsMatchPlugin(PostactionMixin, BaseSolnPlugin):
    name = 'obsmatch'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        self.elementscls = intg.system.elementscls
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')
        if self.fmt == 'primitive':
            self._process = _process_con_to_pri(self.elementscls, self.ndims,
                                                self.cfg)
            all_var_names = list(self.elementscls.privars(self.ndims, self.cfg))
        elif self.fmt == 'conservative':
            self._process = None
            all_var_names = list(self.elementscls.convars(self.ndims, self.cfg))
        else:
            raise ValueError("format must be 'primitive' or 'conservative'")

        obs_vars_cfg = None
        if self.cfg.hasopt(cfgsect, 'observation-variables'):
            vstr = self.cfg.get(cfgsect, 'observation-variables')
            obs_vars_cfg = [v.strip() for v in vstr.replace(',', ' ').split() if v.strip()]
            if not obs_vars_cfg:
                obs_vars_cfg = None

        if rank == root:
            pts, targ_vals, obs_vars = self._load_target(obs_vars_cfg)
        else:
            pts = targ_vals = obs_vars = None

        pts, targ_vals, obs_vars = comm.bcast((pts, targ_vals, obs_vars), root=root)
        self.obs_var_names = list(obs_vars)

        try:
            self.var_indices = [all_var_names.index(v) for v in self.obs_var_names]
        except ValueError as err:
            raise ValueError(
                f'Unknown observation variable in {self.obs_var_names}; '
                f'valid choices are {all_var_names}'
            ) from err

        self.sample_points = self._reshape_points(np.asarray(pts, dtype=float))
        self.target_values = np.asarray(targ_vals, dtype=float)
        self.target_obs = self.target_values.reshape(-1)

        if len(self.target_values) != len(self.sample_points):
            raise ValueError('Number of target values does not match number of points')
        if self.target_values.shape[1] != len(self.obs_var_names):
            raise ValueError('Target value columns do not match observation vars')

        self.psampler = PointSampler(intg.system.mesh, self.sample_points)
        self.psampler.configure_with_intg_nvars(intg, self.nvars)

        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 1)
        self.metric = self.cfg.get(cfgsect, 'metric', 'rel-l2')
        self.metric_eps = self.cfg.getfloat(cfgsect, 'metric-eps', 1.0e-12)
        self.tolerance = self.cfg.getfloat(cfgsect, 'tolerance', -1.0)
        self.stop_on_match = self.cfg.getbool(cfgsect, 'stop-on-match', False)
        self.save_best = self.cfg.getbool(cfgsect, 'save-best', False)
        self.save_match = self.cfg.getbool(cfgsect, 'save-match', True)
        self.save_delta = self.cfg.getfloat(cfgsect, 'save-delta', 0.0)
        self.print_every = self.cfg.getint(cfgsect, 'print-every', 0)

        self.best_mismatch = np.inf
        self.best_time = np.nan
        self.saved_best_mismatch = np.inf
        self.saved_best_time = np.nan
        self.last_mismatch = np.nan
        self.last_time = np.nan
        self.nchecks = 0
        self.match_found = False

        self._target_scale = np.linalg.norm(self.target_obs)
        self._history = None
        if rank == root and self.cfg.hasopt(cfgsect, 'history-file'):
            self._history = init_csv(self.cfg, cfgsect, self._history_header,
                                     filekey='history-file')

        self._write_grads = self.cfg.getbool(cfgsect, 'write-gradients', False)
        self._writer = None
        if self.save_best or self.save_match:
            basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
            basename = self.cfg.get(cfgsect, 'basename')
            self._async_timeout = self.cfg.getfloat(cfgsect, 'async-timeout', 0)
            self.fpdtype = intg.backend.fpdtype
            self.fields = list(first(intg.system.ele_map.values()).convars)
            if self._write_grads:
                dims = 'xyz'[:self.ndims]
                self.fields += [f'grad_{f}_{d}' for f in self.fields for d in dims]

            nvars = self.nvars + self._write_grads*(self.nvars*self.ndims)
            shapes = {etype: (nvars, ele.nupts)
                      for etype, ele in intg.system.ele_map.items()}
            eidxs = {etype: intg.system.mesh.eidxs[etype]
                     for etype in intg.system.ele_map}

            self._writer = NativeWriter.from_integrator(intg, basedir, basename,
                                                        'soln')
            self._writer.set_shapes_eidxs(shapes, eidxs)

        if rank == root:
            print('[obsmatch] Loaded target with '
                  f'{len(self.sample_points)} points, variables='
                  f"{', '.join(self.obs_var_names)}, metric={self.metric}, "
                  f'tolerance={self.tolerance}')

    def setup(self, sdata, serialiser):
        if sdata is not None:
            state = np.asarray(sdata, dtype=float).reshape(-1)
            if len(state) >= 6:
                self.best_mismatch = float(state[0])
                self.best_time = float(state[1])
                self.last_mismatch = float(state[2])
                self.last_time = float(state[3])
                self.nchecks = int(round(state[4]))
                self.match_found = bool(round(state[5]))
            if len(state) >= 8:
                self.saved_best_mismatch = float(state[6])
                self.saved_best_time = float(state[7])

        serialiser.register(self.get_serialiser_prefix(), self._serialise_state)

    @property
    def _history_header(self):
        return 't,mismatch,best_mismatch,is_best,is_match,save,nacptsteps'

    def _serialise_state(self):
        return np.array([
            self.best_mismatch,
            self.best_time,
            self.last_mismatch,
            self.last_time,
            float(self.nchecks),
            float(self.match_found),
            self.saved_best_mismatch,
            self.saved_best_time,
        ])

    def _load_target(self, obs_vars):
        path = Path(self.cfg.getpath(self.cfgsect, 'target-file'))
        suffix = path.suffix.lower()

        if suffix == '.npz':
            pts, vals, obs_vars = _load_npz_target(
                path,
                obs_vars,
                self.cfg.get(self.cfgsect, 'target-points-key', 'points'),
                self.cfg.get(self.cfgsect, 'target-obs-key', 'obs'),
                self.cfg.get(self.cfgsect, 'target-values-key', 'uv'),
                self.cfg.get(self.cfgsect, 'target-variables-key', 'variables')
            )
        elif suffix == '.csv':
            pts, vals, obs_vars = _load_csv_target(path, obs_vars)
        else:
            raise ValueError('target-file must be .npz or .csv')

        return pts, vals, obs_vars

    def _reshape_points(self, pts):
        if pts.ndim != 2:
            raise ValueError('Target points must be a 2D array')

        if pts.shape[1] == self.ndims:
            return pts
        elif pts.shape[1] == 2 and self.ndims == 3:
            if not self.cfg.hasopt(self.cfgsect, 'extrude-z'):
                raise ValueError(
                    '3D observation matching from 2D target points requires '
                    'extrude-z'
                )

            z = self.cfg.getfloat(self.cfgsect, 'extrude-z')
            return np.column_stack([pts, np.full(len(pts), z, dtype=float)])
        else:
            raise ValueError(
                f'Target points dimensionality {pts.shape[1]} is incompatible '
                f'with simulation dimensionality {self.ndims}'
            )

    def _compute_metric(self, obs):
        diff = np.asarray(obs, dtype=float) - self.target_obs

        match self.metric:
            case 'rel-l2':
                return np.linalg.norm(diff) / (self._target_scale + self.metric_eps)
            case 'rmse':
                return np.sqrt(np.mean(diff**2))
            case 'maxabs':
                return np.max(np.abs(diff))
            case _:
                raise ValueError("metric must be one of 'rel-l2', 'rmse', 'maxabs'")

    def _prepare_metadata(self, intg):
        comm, rank, root = get_comm_rank_root()

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        if rank == root:
            metadata = {
                **intg.cfgmeta,
                'stats': stats.tostr(),
                'mesh-uuid': intg.mesh_uuid
            }
        else:
            metadata = None

        sdata = intg.serialiser.serialise()
        if rank == root:
            metadata |= sdata

        return metadata

    def _prepare_data(self, intg):
        data = {}

        if self._write_grads:
            soln, grad_soln = intg.soln, intg.grad_soln
        else:
            soln, grad_soln = intg.soln, None

        for idx, etype in enumerate(intg.system.ele_types):
            d = soln[idx].T.astype(self.fpdtype)

            if self._write_grads:
                g = grad_soln[idx].transpose(3, 2, 0, 1)
                g = g.reshape(len(g), self.ndims*self.nvars, -1)
                d = np.hstack([d, g], dtype=self.fpdtype)

            data[etype] = d

        return data

    def _save_state(self, intg, metric, reason):
        if self._writer is None:
            return

        self._writer.probe()

        data = self._prepare_data(intg)
        metadata = self._prepare_metadata(intg)

        callback = lambda fname, t=intg.tcurr, m=metric, r=reason: self._invoke_postaction(
            intg=intg, mesh=intg.system.mesh.fname, soln=fname, t=t,
            mismatch=m, reason=r
        )

        self._writer.write(data, intg.tcurr, metadata, self._async_timeout,
                           callback)

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps:
            return

        comm, rank, root = get_comm_rank_root()
        samples = self.psampler.sample(list(intg.soln), process=self._process)

        if rank == root:
            values = np.asarray(samples[:, self.var_indices], dtype=float)
            obs = values.reshape(-1)
            metric = self._compute_metric(obs)

            prev_best = self.best_mismatch
            is_best = metric < prev_best
            new_match = self.tolerance > 0 and metric <= self.tolerance and not self.match_found

            self.last_mismatch = metric
            self.last_time = intg.tcurr
            self.nchecks += 1

            if is_best:
                self.best_mismatch = metric
                self.best_time = intg.tcurr

            self.match_found = self.match_found or (self.tolerance > 0 and metric <= self.tolerance)

            save = False
            reason = ''
            if new_match and self.save_match:
                save = True
                reason = 'match'
            elif is_best and self.save_best and (
                np.isinf(self.saved_best_mismatch)
                or self.saved_best_mismatch - metric > self.save_delta
            ):
                save = True
                reason = 'best'

            if save and is_best:
                self.saved_best_mismatch = metric
                self.saved_best_time = intg.tcurr

            abort = bool(new_match and self.stop_on_match)

            if self._history is not None:
                self._history(
                    intg.tcurr, metric, self.best_mismatch, int(is_best),
                    int(new_match), reason, intg.nacptsteps
                )

            if self.print_every > 0 and (self.nchecks % self.print_every == 0 or save):
                msg = (f'[obsmatch] t={intg.tcurr:.8f} mismatch={metric:.8e} '
                       f'best={self.best_mismatch:.8e}')
                if save:
                    msg += f' save={reason}'
                if new_match:
                    msg += ' matched'
                print(msg)

            payload = {
                'metric': metric,
                'best_mismatch': self.best_mismatch,
                'best_time': self.best_time,
                'last_mismatch': self.last_mismatch,
                'last_time': self.last_time,
                'nchecks': self.nchecks,
                'match_found': self.match_found,
                'saved_best_mismatch': self.saved_best_mismatch,
                'saved_best_time': self.saved_best_time,
                'save': save,
                'save_reason': reason,
                'abort': abort,
                'new_match': new_match,
            }
        else:
            payload = None

        payload = comm.bcast(payload, root=root)

        self.best_mismatch = payload['best_mismatch']
        self.best_time = payload['best_time']
        self.last_mismatch = payload['last_mismatch']
        self.last_time = payload['last_time']
        self.nchecks = payload['nchecks']
        self.match_found = payload['match_found']
        self.saved_best_mismatch = payload['saved_best_mismatch']
        self.saved_best_time = payload['saved_best_time']

        if payload['save']:
            self._save_state(intg, payload['metric'], payload['save_reason'])

        if payload['abort']:
            intg.plugin_abort(
                f'[obsmatch] match threshold reached at t={intg.tcurr:.8f} '
                f'with mismatch={payload["metric"]:.8e}'
            )

    def finalise(self, intg):
        super().finalise(intg)

        if self._writer is not None:
            self._writer.flush()

        if self._history is not None:
            self._history.outf.flush()
            self._history.outf.close()
