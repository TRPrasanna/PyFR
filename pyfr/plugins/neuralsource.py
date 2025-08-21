# pyfr/plugins/neuralsource.py
import math
import numpy as np

from pyfr.plugins.base import BaseSolverPlugin


class NeuralSourcePlugin(BaseSolverPlugin):
    """
    DRL-driven volumetric body forces along three line segments.
    Uses the first three entries of env.current_control as coefficients.
    Each coefficient is linearly ramped over the action interval.

    Kernel builds Sy = c1*mask1 + c2*mask2 + c3*mask3 and adds to rhov, and to E via u·S.
    Sx is zero here by design.

    Config section: [solver-plugin-neuralsource]
      Required:
        mask1, mask2, mask3 : expressions defining the three line masks
      Optional:
        action-interval      : fallback if no RL env is present
    """
    name = 'neuralsource'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    _NCHAN = 3  # three control channels

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)
        self.intg = intg
        self.sys = intg.system
        self.backend = intg.backend

        # Action interval: prefer RL env value
        env = getattr(self.sys, 'env', None)
        if env is not None and hasattr(env, 'action_interval'):
            act_dt = float(env.action_interval)
        else:
            act_dt = self.cfg.getfloat(cfgsect, 'action-interval', 0.1)

        # Extern buffers shared with element kernels:
        # control_params_src: shape [3 x 3], rows are channels 0..2, cols [Q0, Q1, t0]
        self.control_params = self.backend.matrix((self._NCHAN, 3))
        self.t_act_interval = self.backend.matrix((1, 1))

        self.control_params.set(np.zeros((self._NCHAN, 3), dtype=np.float64))
        self.t_act_interval.set(np.array([[act_dt]], dtype=np.float64))

        # Bind externs
        for _, eles in self.sys.ele_map.items():
            eles._set_external('control_params_src',
                               'broadcast fpdtype_t[{}][3]'.format(self._NCHAN),
                               value=self.control_params)
            eles._set_external('t_act_interval_src',
                               'broadcast fpdtype_t[1][1]',
                               value=self.t_act_interval)

        # Build masks from cfg expressions
        subs = self.cfg.items('constants')
        subs |= self.cfg.items(cfgsect)
        subs |= dict(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs |= dict(abs='fabs', pi=math.pi)

        convars = self.sys.elementscls.convarmap[self.sys.ndims]
        subs |= {v: f'u[{i}]' for i, v in enumerate(convars)}

        for key in ('mask1', 'mask2', 'mask3'):
            if not self.cfg.hasopt(cfgsect, key):
                raise ValueError(f"[solver-plugin-neuralsource] requires '{key}'")

        mask1 = self.cfg.getexpr(cfgsect, 'mask1', subs=subs)
        mask2 = self.cfg.getexpr(cfgsect, 'mask2', subs=subs)
        mask3 = self.cfg.getexpr(cfgsect, 'mask3', subs=subs)

        tplargs = dict(
            mask1=mask1,
            mask2=mask2,
            mask3=mask3,
            ndims=self.sys.ndims,
            nvars=len(convars),
            nch=self._NCHAN,
        )

        # Register the macro on all element types
        for _, eles in self.sys.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.neuralsource',
                               'neuralsource',
                               tplargs,
                               ploc=True,
                               soln=True)

        self._last_step_count = -1

    def __call__(self, intg):
        """Update ramps when the env advances to a new action."""
        env = getattr(self.sys, 'env', None)
        if env is None:
            return

        sc = getattr(env, 'step_count', -1)
        if sc != self._last_step_count:
            cc = np.asarray(env.current_control, dtype=np.float64)
            pc = np.asarray(env.previous_control, dtype=np.float64)
            if cc.size < self._NCHAN or pc.size < self._NCHAN:
                raise RuntimeError(
                    f"neuralsource expects at least {self._NCHAN} action channels; "
                    f"got current={cc.size}, previous={pc.size}"
                )

            t0 = float(env.current_time)
            rows = []
            for k in range(self._NCHAN):
                rows.append([pc[k], cc[k], t0])
            self.control_params.set(np.array(rows, dtype=np.float64))

            self._last_step_count = sc
