import ast
import math
import operator

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins._surface import cross_fluxpts
from pyfr.plugins.soln.fluidforce import FluidForceIntegrator
from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.points import PointSampler


def _integrate_trapezoid(y, x):
    # NumPy 2.x removed np.trapz in favour of np.trapezoid.
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x)
    else:
        return np.trapz(y, x=x)


class ReinforcementLearningPlugin(BaseSolverPlugin):
    name = 'reinforcementlearning'
    systems = 'navier-stokes'
    formulations = ['std']
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Import torch lazily so non-RL runs do not load PyTorch libraries.
        import torch

        self._torch = torch

        comm, rank, root = get_comm_rank_root()

        self.device = self._torch.device('cpu')
        self.elementscls = intg.system.elementscls

        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')

        spts = self.cfg.get(cfgsect, 'probe-pts')
        if ',' in spts:
            spts = self.cfg.getliteral(cfgsect, 'probe-pts')

        self.psampler = PointSampler(intg.system.mesh, spts)
        self.psampler.configure_with_intg_nvars(intg, self.nvars)

        default_var_list = ['u', 'v', 'p']
        var_string = self.cfg.get(
            cfgsect, 'observation-variables', ','.join(default_var_list)
        )
        self.obs_var_names = [v.strip() for v in var_string.replace(',', ' ').split()]

        primitive_names = list(self.elementscls.privars(self.ndims, self.cfg))
        try:
            self.var_indices = [primitive_names.index(v) for v in self.obs_var_names]
        except ValueError as err:
            raise ValueError(
                '[reinforcementlearning] observation-variables: '
                f'unknown name in {self.obs_var_names}; '
                f'valid choices: {primitive_names}'
            ) from err

        self.observation_size = len(self.psampler.pts) * len(self.var_indices)

        self.action_interval = self.cfg.getfloat(cfgsect, 'action-interval', 0.1)
        self.last_action_time = intg.tcurr

        self._viscous = 'navier-stokes' in intg.system.name
        self._ac = intg.system.name.startswith('ac')
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')
        self._constants = self.cfg.items_as('constants', float)
        if self.cfg.hasopt(cfgsect, 'viscous-grad-source'):
            vsrc = self.cfg.get(cfgsect, 'viscous-grad-source')
            if vsrc != 'corrected':
                raise ValueError(
                    "viscous-grad-source='local' is no longer supported; "
                    "use corrected gradients"
                )

        mcomp = 3 if self.ndims == 3 else 1
        self._mcomp = mcomp if self.cfg.hasopt(cfgsect, 'morigin') else 0
        morigin = None
        if self._mcomp:
            morigin = np.array(self.cfg.getliteral(cfgsect, 'morigin'))
            if len(morigin) != self.ndims:
                raise ValueError(f'morigin must have {self.ndims} components')

        self.surf_bnames = self.cfg.getliteral(cfgsect, 'surfaces')
        if not isinstance(self.surf_bnames, list):
            self.surf_bnames = [self.surf_bnames]
        if not self.surf_bnames:
            raise ValueError('No surfaces specified for forces/moment calculation')

        self.ff_int = {}
        for surf in self.surf_bnames:
            bcranks = comm.gather(surf in intg.system.mesh.bcon, root=root)

            if rank == root and not any(bcranks):
                raise RuntimeError(f'Boundary {surf} does not exist')

            self.ff_int[surf] = FluidForceIntegrator(
                self.cfg, cfgsect, intg.system, surf, morigin
            )

        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 10)

        self.force_times = []
        self.drag_history = []
        self.lift_history = []
        self.moment_history = []
        self.action_history = []
        self.avg_window = self.cfg.getfloat(cfgsect, 'averaging-window', 0.5)

        self.reward_function = self.cfg.get(
            cfgsect, 'reward-function', '-abs(avg_moment)'
        )

        self.normalize_reward = self.cfg.getbool(cfgsect, 'normalize-reward', False)
        self.ref_values = {}
        if self.normalize_reward:
            for key in ['drag-ref', 'lift-ref', 'moment-ref', 'action-ref']:
                if self.cfg.hasopt(cfgsect, key):
                    self.ref_values[key.replace('-ref', '')] = self.cfg.getfloat(
                        cfgsect, key
                    )

        self.expr_evaluator = SafeExpressionEvaluator()
        self.used_variables = self.expr_evaluator.find_used_variables(
            self.reward_function
        )

    def _control_value(self, intg):
        env = getattr(intg, 'env', None) or getattr(intg.system, 'env', None)
        if env is None:
            return np.zeros(1, dtype=np.float64)

        prev = np.asarray(getattr(env, 'previous_control', 0.0), dtype=np.float64)
        curr = np.asarray(getattr(env, 'current_control', prev), dtype=np.float64)

        act_dt = float(getattr(env, 'action_interval', self.action_interval))
        t0 = float(getattr(env, 'current_time', intg.tcurr))

        if act_dt <= 0:
            return curr

        control = (curr - prev) / act_dt * (intg.tcurr - t0) + prev
        lower = np.minimum(prev, curr)
        upper = np.maximum(prev, curr)

        return np.maximum(lower, np.minimum(control, upper))

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps:
            return

        control = self._control_value(intg)
        self.sumabsact = np.sum(control**2) + np.sum(control)**2

        comm, rank, root = get_comm_rank_root()
        fm = self._compute_fm(intg, dict(zip(intg.system.ele_types, intg.soln)))

        if self._viscous:
            pidx, vidx, midx = 0, 1, 2
        else:
            pidx, vidx, midx = 0, None, 1

        if rank == root:
            t = intg.tcurr

            drag = (fm[pidx, 0] + (fm[vidx, 0] if vidx is not None else 0.0)
                    + fm[midx, 0]) * 2
            lift = (fm[pidx, 1] + (fm[vidx, 1] if vidx is not None else 0.0)
                    + fm[midx, 1]) * 2

            if self._mcomp:
                moment = (fm[pidx, 2]
                          + (fm[vidx, 2] if vidx is not None else 0.0)
                          + fm[midx, 2]) * 2

            self.force_times.append(t)
            self.drag_history.append(drag)
            self.lift_history.append(lift)
            if self._mcomp:
                self.moment_history.append(moment)
            self.action_history.append(self.sumabsact)

            while self.force_times and self.force_times[0] < t - self.avg_window:
                self.force_times.pop(0)
                self.drag_history.pop(0)
                self.lift_history.pop(0)
                self.action_history.pop(0)
                if self._mcomp:
                    self.moment_history.pop(0)

    def _compute_fm(self, intg, solns):
        comm, rank, root = get_comm_rank_root()

        ndims = self.ndims
        mcomp = self._mcomp

        if self._viscous:
            pidx, vidx, midx = 0, 1, 2
            fm = np.zeros((3, ndims + mcomp))
            grads = dict(zip(intg.system.ele_types, intg.grad_soln))
        else:
            pidx, vidx, midx = 0, None, 1
            fm = np.zeros((2, ndims + mcomp))
            grads = None

        for surf in self.surf_bnames:
            ff_int = self.ff_int[surf]

            for (etype, fidx), m0 in ff_int.m0.items():
                nfpts, nupts = m0.shape

                uupts = solns[etype][..., ff_int.eidxs[etype, fidx]]

                ufpts = m0 @ uupts.reshape(nupts, -1)
                ufpts = ufpts.reshape(nfpts, self.nvars, -1)
                ufpts = ufpts.swapaxes(0, 1)

                pri_vars = self.elementscls.con_to_pri(ufpts, self.cfg)
                p = pri_vars[0 if self._ac else -1]

                qwts = ff_int.qwts[etype, fidx]
                norms = ff_int.norms[etype, fidx]

                pforce = p[None, :, :]*norms
                fm[pidx, :ndims] += np.einsum('f,dfe->d', qwts, pforce)

                # Momentum flux contribution
                vs = np.array(pri_vars[1:-1])
                rho = np.ones_like(vs[0]) if self._ac else pri_vars[0]
                rhovs = rho[None, :, :] * vs
                rhovn = np.einsum('dfe,dfe->fe', rhovs, norms)
                momflux = rhovn[None, :, :]*vs
                fm[midx, :ndims] += np.einsum('f,dfe->d', qwts, momflux)

                if self._viscous:
                    duupts = grads[etype][..., ff_int.eidxs[etype, fidx]]
                    duupts = duupts.reshape(ndims, nupts, -1)

                    dufpts = np.array([m0 @ du for du in duupts])
                    dufpts = dufpts.reshape(ndims, nfpts, self.nvars, -1)
                    dufpts = dufpts.swapaxes(1, 2)

                    if self._ac:
                        vis = self.ac_stress_tensor(dufpts)
                    else:
                        vis = self.stress_tensor(ufpts, dufpts)

                    viscf = np.einsum('dkfe,kfe->dfe', vis, norms)
                    fm[vidx, :ndims] += np.einsum('f,dfe->d', qwts, viscf)

                if self._mcomp:
                    rfpts = ff_int.rfpts[etype, fidx]

                    rcf = cross_fluxpts(rfpts, pforce)
                    fm[pidx, ndims:] += np.einsum('f,mfe->m', qwts, rcf)

                    rcf = cross_fluxpts(rfpts, momflux)
                    fm[midx, ndims:] += np.einsum('f,mfe->m', qwts, rcf)

                    if self._viscous:
                        rcf = cross_fluxpts(rfpts, viscf)
                        fm[vidx, ndims:] += np.einsum('f,mfe->m', qwts, rcf)

        if rank != root:
            comm.Reduce(fm, None, op=mpi.SUM, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, fm, op=mpi.SUM, root=root)

        return fm

    def _get_observation(self, solver):
        comm, rank, root = get_comm_rank_root()

        samps = self.psampler.sample(list(solver.soln))

        if rank == root:
            if samps is None:
                obs = np.zeros(self.observation_size, dtype=np.float32)
            else:
                samps = np.array(samps)

                if self.fmt == 'primitive' and samps.size:
                    samps = self.elementscls.con_to_pri(samps.T, self.cfg)
                    samps = np.array(samps).T

                if samps.size:
                    samps = samps[:, self.var_indices]

                obs = np.asarray(samps, dtype=np.float32).reshape(-1)

                if obs.size != self.observation_size:
                    obs = np.resize(obs, self.observation_size)
        else:
            obs = None

        obs = comm.bcast(obs, root=root)

        return self._torch.tensor(obs, device=self.device).float()

    def _compute_std(self, history, mean):
        if len(history) <= 1 or len(self.force_times) <= 1:
            return 0.0

        sq_diffs = [(x - mean)**2 for x in history]
        variance = _integrate_trapezoid(sq_diffs, self.force_times) / (
            self.force_times[-1] - self.force_times[0]
        )

        return np.sqrt(variance)

    def _get_reward(self, solver):
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            if not self.force_times:
                reward = 0.0
            else:
                variables = {}

                needs_avg_drag = 'avg_drag' in self.used_variables
                needs_avg_lift = 'avg_lift' in self.used_variables
                needs_avg_moment = 'avg_moment' in self.used_variables
                needs_avg_sumabsact = 'avg_sumabsact' in self.used_variables

                needs_std_drag = 'std_drag' in self.used_variables
                needs_std_lift = 'std_lift' in self.used_variables
                needs_std_moment = 'std_moment' in self.used_variables

                delta_t = self.force_times[-1] - self.force_times[0]
                if delta_t <= 0:
                    reward = 0.0
                else:
                    if needs_avg_drag:
                        variables['avg_drag'] = (
                            _integrate_trapezoid(self.drag_history,
                                                 self.force_times)
                            / delta_t
                        )

                    if needs_avg_lift:
                        variables['avg_lift'] = (
                            _integrate_trapezoid(self.lift_history,
                                                 self.force_times)
                            / delta_t
                        )

                    if needs_avg_moment and self._mcomp and self.moment_history:
                        variables['avg_moment'] = (
                            _integrate_trapezoid(self.moment_history,
                                                 self.force_times)
                            / delta_t
                        )
                    elif needs_avg_moment:
                        variables['avg_moment'] = 0.0

                    if needs_avg_sumabsact:
                        variables['avg_sumabsact'] = (
                            _integrate_trapezoid(self.action_history,
                                                 self.force_times)
                            / delta_t
                        )

                    if needs_std_drag:
                        avg_drag = variables.get(
                            'avg_drag',
                            _integrate_trapezoid(self.drag_history,
                                                 self.force_times)
                            / delta_t
                        )
                        variables['std_drag'] = self._compute_std(
                            self.drag_history, avg_drag
                        )

                    if needs_std_lift:
                        avg_lift = variables.get(
                            'avg_lift',
                            _integrate_trapezoid(self.lift_history,
                                                 self.force_times)
                            / delta_t
                        )
                        variables['std_lift'] = self._compute_std(
                            self.lift_history, avg_lift
                        )

                    if needs_std_moment and self._mcomp and self.moment_history:
                        avg_moment = variables.get(
                            'avg_moment',
                            _integrate_trapezoid(self.moment_history,
                                                 self.force_times)
                            / delta_t
                        )
                        variables['std_moment'] = self._compute_std(
                            self.moment_history, avg_moment
                        )
                    elif needs_std_moment:
                        variables['std_moment'] = 0.0

                    if self.normalize_reward:
                        for var_name, value in list(variables.items()):
                            base_name = var_name.split('_')[1]
                            if (base_name in self.ref_values and
                                    self.ref_values[base_name] != 0):
                                variables[var_name] = value / self.ref_values[base_name]

                    try:
                        reward = float(
                            self.expr_evaluator.evaluate(self.reward_function, variables)
                        )
                    except Exception as e:
                        print(f'Error evaluating reward function: {e}')
                        reward = -1000.0
        else:
            reward = None

        return comm.bcast(reward, root=root)

    def reset(self):
        self.force_times.clear()
        self.drag_history.clear()
        self.lift_history.clear()
        self.moment_history.clear()
        self.action_history.clear()
        self.sumabsact = 0.0

    def stress_tensor(self, u, du):
        c = self._constants

        rho, E = u[0], u[-1]

        gradrho, gradrhou = du[:, 0], du[:, 1:-1]

        gradu = (gradrhou - gradrho[:, None]*u[None, 1:-1]/rho) / rho

        bulk = np.eye(self.ndims)[:, :, None, None]*np.trace(gradu)

        mu = c['mu']

        if self._viscorr == 'sutherland':
            cpT = c['gamma']*(E/rho - 0.5*np.sum(u[1:-1]**2, axis=0)/rho**2)
            Trat = cpT/c['cpTref']
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        return -mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)

    def ac_stress_tensor(self, du):
        gradu, nu = du[:, 1:], self._constants['nu']

        return -nu*(gradu + gradu.swapaxes(0, 1))


class SafeExpressionEvaluator:
    """
    Evaluate a restricted mathematical expression safely.

    Allowed:
        * numeric literals
        * variables passed in `variables`
        * +  -  *  /  **  unary +/-
        * abs, sqrt, exp, log, log10, sin, cos, tan, min, max
        * constants: pi, e
    """

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    _FUNCS = {
        'abs': abs,
        'sqrt': math.sqrt,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'min': min,
        'max': max,
    }

    _CONST = {'pi': math.pi, 'e': math.e}

    def evaluate(self, expr: str, variables: dict | None = None) -> float:
        variables = variables or {}

        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError as err:
            raise ValueError(f'Invalid expression: {err}') from None

        return float(self._eval(tree.body, variables))

    def find_used_variables(self, expr: str) -> set[str]:
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            return set()

        vars_ = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in self._FUNCS and node.id not in self._CONST:
                    vars_.add(node.id)

        return vars_

    def _eval(self, node: ast.AST, env: dict[str, float]) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError('Only numeric literals are allowed')

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self._CONST:
                return self._CONST[node.id]
            raise ValueError(f'Unknown variable: {node.id}')

        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.operand, env))

        if isinstance(node, ast.BinOp) and type(node.op) in self._OPS:
            left = self._eval(node.left, env)
            right = self._eval(node.right, env)
            return self._OPS[type(node.op)](left, right)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname not in self._FUNCS:
                raise ValueError(f'Function not allowed: {fname}')

            args = [self._eval(arg, env) for arg in node.args]
            return self._FUNCS[fname](*args)

        raise ValueError(
            f'Unsupported expression element: '
            f'{ast.dump(node, annotate_fields=False)}'
        )
