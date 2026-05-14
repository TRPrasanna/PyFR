from ast import literal_eval

import numpy as np

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)
from pyfr.solvers.euler.inters import MassFlowBCMixin, PressureBCMixin


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, visc_corr=visc_corr,
                             shock_capturing=shock_capturing, c=self.c,
                             p_min=self.p_min)


class NavierStokesIntInters(TplargsMixin,
                            BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self.scal_lhs, urin=self.scal_rhs,
            ulout=self._comm_lhs, urout=self._comm_rhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs
        )


class NavierStokesMPIInters(TplargsMixin,
                            BaseAdvectionDiffusionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self.scal_lhs, urin=self.scal_rhs, ulout=self._comm_lhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs
        )


class NavierStokesBaseBCInters(TplargsMixin, BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional BC specific template arguments
        self._tplargs['bctype'] = self.type
        self._tplargs['bccfluxstate'] = self.cflux_state

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self.scal_lhs,
            ulout=self._comm_lhs, nlin=self._pnorm_lhs,
            **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self.scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs,
            artvisc=self.artvisc, **self._external_vals
        )

    def comm_entropy_kernel(self, entmin_lhs):
        # Physics-specific callback for entropy filtering
        self._be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.bccent'
        )

        return lambda: self._be.kernel(
            'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, entmin_lhs=entmin_lhs,
            nl=self._pnorm_lhs, ul=self.scal_lhs, **self._external_vals
        )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost-imperm'


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self.c['cpTt'], = self._eval_opts(['cpTt'])
        self.c['pt'], = self._eval_opts(['pt'])
        self.c['Rdcp'] = (gamma - 1.0)/gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0]*np.pi/180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self.c['vc'] = velcomps[:self.ndims]


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(['p'], lhs)


class NavierStokesCharRiemInvMassFlowBCInters(MassFlowBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-mass-flow'
    cflux_state = 'ghost'


class NavierStokesCharRiemInvPressureBCInters(PressureBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-pressure'
    cflux_state = 'ghost'


class NavierStokesAdiaJetBCInters(NavierStokesBaseBCInters):
    type = 'adia-jet'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )


class _AdiaJetRLControlMixin:
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Action interval used for linear interpolation in the kernel.
        self.t_act_interval = be.matrix((1, 1))
        self.set_external('t_act_interval', 'broadcast fpdtype_t[1][1]',
                          value=self.t_act_interval)
        self.t_act_interval.set(np.array(
            [[cfg.getfloat('solver-plugin-reinforcementlearning',
                           'action-interval')]]
        ))

        self.control_params = be.matrix((1, 3))
        self.set_external('control_params', 'broadcast fpdtype_t[1][3]',
                          value=self.control_params)
        self._control_params_host = np.array([[0.0, 0.0, 0.0]])
        self.control_params.set(self._control_params_host)

        self._current_target = 0.0
        self._last_env_step = None

    def setup(self, sdata, prevcfg):
        sect_eq = (prevcfg is not None and
                   self.cfg.sect_eq(prevcfg, self.cfgsect))

        if sdata is not None and sect_eq:
            params = np.asarray(sdata, dtype=np.float64).reshape(1, 3)
            self._control_params_host = params.copy()
            self.control_params.set(params)
            self._current_target = float(params[0, 1])

    @classmethod
    def serialisefn(cls, bciface, prefix, srl):
        sfn = lambda: bciface._control_params_host.copy()
        srl.register(prefix, sfn if bciface else None)

    @classmethod
    def preparefn(cls, bciface, mesh, elemap):
        if bciface:
            return bciface.prepare
        else:
            return None

    def _target_from_env(self, env):
        raise NotImplementedError

    def prepare(self, system, ubank, t, kerns):
        env = getattr(system, 'env', None)
        if env is None:
            return

        env_step = int(getattr(env, 'step_count', -1))
        if env_step == self._last_env_step:
            return

        target = float(self._target_from_env(env))
        params = np.array([[self._current_target, target, t]])
        self.control_params.set(params)
        self._control_params_host = params
        self._current_target = target
        self._last_env_step = env_step

    def seed_control_state_from_env(self, env, t):
        target = float(self._target_from_env(env))
        params = np.array([[target, target, t]])
        self.control_params.set(params)
        self._control_params_host = params
        self._current_target = target
        self._last_env_step = None


class NavierStokesAdiaJetNeuralType5BCInters(_AdiaJetRLControlMixin,
                                             NavierStokesBaseBCInters):
    type = 'adia-jet-neural-type5'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        self.actuator_id = cfg.getint(cfgsect, 'actuator-number')

    def _target_from_env(self, env):
        ctrl = np.asarray(getattr(env, 'current_control', 0.0), dtype=np.float64)
        return ctrl[self.actuator_id]


class NavierStokesAdiaJetNeuralType5ResidualBCInters(_AdiaJetRLControlMixin,
                                                     NavierStokesBaseBCInters):
    type = 'adia-jet-neural-type5-residual'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

    def _target_from_env(self, env):
        ctrl = np.asarray(getattr(env, 'current_control', 0.0), dtype=np.float64)
        return -np.sum(ctrl)


class _AdiaJetRLMultiControlMixin:
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        if cfg.hasopt(cfgsect, 'num-actuators'):
            self.num_actuators = cfg.getint(cfgsect, 'num-actuators')
        else:
            self.num_actuators = cfg.getint(
                'solver-plugin-reinforcementlearning', 'num-control-actions'
            )

        if self.num_actuators < 1:
            raise ValueError('num-actuators must be >= 1')

        # Compile-time template arg for unrolling actuator contributions.
        self._tplargs['nctrl'] = self.num_actuators

        # Action interval used for linear interpolation in the kernel.
        self.t_act_interval = be.matrix((1, 1))
        self.set_external('t_act_interval', 'broadcast fpdtype_t[1][1]',
                          value=self.t_act_interval)
        self.t_act_interval.set(np.array(
            [[cfg.getfloat('solver-plugin-reinforcementlearning',
                           'action-interval')]]
        ))

        self.control_params = be.matrix((self.num_actuators, 3))
        self.set_external(
            'control_params',
            f'broadcast fpdtype_t[{self.num_actuators}][3]',
            value=self.control_params
        )
        self._control_params_host = np.zeros((self.num_actuators, 3))
        self.control_params.set(self._control_params_host)

        # Per-actuator spatial mask bounds:
        #   <axis>-min<i>, <axis>-max<i>
        # with aliases:
        #   <axis><i>-min, <axis><i>-max
        # where axis in {x, y, z} for the active dimensions.
        self.actuator_bounds = be.matrix((self.num_actuators, 2*self.ndims))
        self.set_external(
            'actuator_bounds',
            f'broadcast fpdtype_t[{self.num_actuators}][{2*self.ndims}]',
            value=self.actuator_bounds
        )
        self.actuator_bounds.set(self._read_actuator_bounds(cfgsect, cfg))

        # type6 always references ploc in the kernel for masking.
        if 'ploc' not in self._external_args:
            spec = f'in fpdtype_t[{self.ndims}]'
            value = self._const_mat(lhs, 'get_ploc_for_inters')
            self.set_external('ploc', spec, value=value)

        self._current_targets = np.zeros(self.num_actuators)
        self._last_env_step = None

    def setup(self, sdata, prevcfg):
        sect_eq = (prevcfg is not None and
                   self.cfg.sect_eq(prevcfg, self.cfgsect))

        if sdata is not None and sect_eq:
            params = np.asarray(sdata, dtype=np.float64).reshape(
                self.num_actuators, 3
            )
            self._control_params_host = params.copy()
            self.control_params.set(params)
            self._current_targets = params[:, 1].copy()

    @classmethod
    def serialisefn(cls, bciface, prefix, srl):
        def sfn():
            return bciface._control_params_host.copy()

        srl.register(prefix, sfn if bciface else None)

    @classmethod
    def preparefn(cls, bciface, mesh, elemap):
        if bciface:
            return bciface.prepare
        else:
            return None

    def _targets_from_env(self, env):
        ctrl = np.asarray(getattr(env, 'current_control', 0.0), dtype=np.float64)
        ctrl = np.atleast_1d(ctrl).ravel()

        if ctrl.size < self.num_actuators:
            raise ValueError(
                f'Environment has {ctrl.size} control actions, but '
                f'{self.num_actuators} are required by {self.cfgsect}'
            )

        return ctrl[:self.num_actuators]

    def _read_actuator_bounds(self, cfgsect, cfg):
        bounds = np.empty((self.num_actuators, 2*self.ndims), dtype=np.float64)
        dnames = 'xyz'[:self.ndims]
        inf = 1.0e100

        for i in range(self.num_actuators):
            for d, dn in enumerate(dnames):
                lo = self._get_bound_opt(cfgsect, cfg, dn, i, 'min', -inf)
                hi = self._get_bound_opt(cfgsect, cfg, dn, i, 'max', inf)

                if lo > hi:
                    raise ValueError(
                        f'Invalid bounds for actuator {i} axis {dn}: '
                        f'{lo} > {hi}'
                    )

                bounds[i, 2*d] = lo
                bounds[i, 2*d + 1] = hi

        return bounds

    def _get_bound_opt(self, cfgsect, cfg, axis, aid, side, default):
        # Preferred: x-min0 / x-max0
        # Alias:     x0-min / x0-max
        key1 = f'{axis}-{side}{aid}'
        key2 = f'{axis}{aid}-{side}'

        if cfg.hasopt(cfgsect, key1):
            return cfg.getfloat(cfgsect, key1)
        if cfg.hasopt(cfgsect, key2):
            return cfg.getfloat(cfgsect, key2)
        return default

    def prepare(self, system, ubank, t, kerns):
        env = getattr(system, 'env', None)
        if env is None:
            return

        env_step = int(getattr(env, 'step_count', -1))
        if env_step == self._last_env_step:
            return

        targets = self._targets_from_env(env)

        params = np.empty((self.num_actuators, 3))
        params[:, 0] = self._current_targets
        params[:, 1] = targets
        params[:, 2] = t

        self.control_params.set(params)
        self._control_params_host = params
        self._current_targets = targets.copy()
        self._last_env_step = env_step

    def seed_control_state_from_env(self, env, t):
        targets = self._targets_from_env(env)
        params = np.empty((self.num_actuators, 3), dtype=np.float64)
        params[:, 0] = targets
        params[:, 1] = targets
        params[:, 2] = t

        self.control_params.set(params)
        self._control_params_host = params
        self._current_targets = targets.copy()
        self._last_env_step = None


class _AdiaJetRLMultiSlotControlMixin(_AdiaJetRLMultiControlMixin):
    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.num_z_slots, z_slot_bounds = self._read_z_slot_bounds(cfgsect, cfg)
        self._tplargs['nzslots'] = self.num_z_slots

        # Optional shared spanwise slot mask for repeated slots controlled by
        # the same actuator action.  Keep one dummy slot when disabled so the
        # generated kernel signature remains simple.
        self.z_slot_bounds = be.matrix((max(self.num_z_slots, 1), 2))
        self.set_external(
            'z_slot_bounds',
            f'broadcast fpdtype_t[{max(self.num_z_slots, 1)}][2]',
            value=self.z_slot_bounds
        )
        self.z_slot_bounds.set(z_slot_bounds)

    def _read_z_slot_bounds(self, cfgsect, cfg):
        if self.ndims < 3 or not cfg.hasopt(cfgsect, 'z-slots'):
            return 0, np.array([[-1.0e100, 1.0e100]], dtype=np.float64)

        slots = np.asarray(literal_eval(cfg.get(cfgsect, 'z-slots')),
                           dtype=np.float64)

        if slots.ndim != 2 or slots.shape[1] != 2:
            raise ValueError(
                f'{cfgsect}: z-slots must be a list of (zmin, zmax) pairs'
            )

        if len(slots) < 1:
            raise ValueError(f'{cfgsect}: z-slots must not be empty')

        for i, (lo, hi) in enumerate(slots):
            if lo > hi:
                raise ValueError(
                    f'Invalid z-slots entry {i} in {cfgsect}: {lo} > {hi}'
                )

        return len(slots), slots


class NavierStokesAdiaJetNeuralType6BCInters(_AdiaJetRLMultiControlMixin,
                                             NavierStokesBaseBCInters):
    type = 'adia-jet-neural-type6'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        comps = ['u', 'v', 'w'][:self.ndims]
        expr_keys = []
        defaults = {}

        # Each actuator i takes ui(x,y,...) and vi(x,y,...) (and wi for 3-D).
        for i in range(self.num_actuators):
            for comp in comps:
                key = f'{comp}{i}'
                expr_keys.append(key)
                defaults[key] = 0

        self.c |= self._exp_opts(expr_keys, lhs, default=defaults)


class NavierStokesAdiaJetNeuralType7BCInters(_AdiaJetRLMultiSlotControlMixin,
                                             NavierStokesBaseBCInters):
    type = 'adia-jet-neural-type7'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        comps = ['u', 'v', 'w'][:self.ndims]
        expr_keys = []
        defaults = {}

        # Each actuator i takes ui(x,y,...) and vi(x,y,...) (and wi for 3-D).
        for i in range(self.num_actuators):
            for comp in comps:
                key = f'{comp}{i}'
                expr_keys.append(key)
                defaults[key] = 0

        self.c |= self._exp_opts(expr_keys, lhs, default=defaults)
