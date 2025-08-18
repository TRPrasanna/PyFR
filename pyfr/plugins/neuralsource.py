# pyfr/plugins/neuralsource.py
import math
import numpy as np

from pyfr.plugins.base import BaseSolverPlugin


class NeuralSourcePlugin(BaseSolverPlugin):
    """
    Volume-source controller driven by the first two DRL action channels.

    Exposes two time-ramped coefficients c1(t), c2(t) to element kernels via
    extern buffers. The Mako macro then builds sources like:
        Sx = c1 * mask1 + c2 * mask2
        Sy = optional (same pattern) or 0

    Config section: [solver-plugin-neuralsource]

    Required keys in that section:
        - mask1 : expression for first line mask (dimensionless)
        - mask2 : expression for second line mask
      Optional:
        - mask1y, mask2y : if you also want a Sy contribution
        - action-interval : fallback action interval if no RL env is present
    """
    name = 'neuralsource'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)
        self.intg = intg
        self.sys = intg.system
        self.backend = intg.backend

        # Action interval: prefer the RL env value, fallback to cfg
        env = getattr(self.sys, 'env', None)
        if env is not None and hasattr(env, 'action_interval'):
            act_dt = float(env.action_interval)
        else:
            act_dt = self.cfg.getfloat(cfgsect, 'action-interval', 0.1)

        # Extern buffers shared with element kernels
        # control_params_src: shape [2 x 3], rows are channels 0 and 1, cols are [Q0, Q1, t0]
        self.control_params = self.backend.matrix((2, 3))
        self.t_act_interval = self.backend.matrix((1, 1))

        self.control_params.set(np.array([[0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0]], dtype=np.float64))
        self.t_act_interval.set(np.array([[act_dt]], dtype=np.float64))

        # Bind externs so the macro can read them
        for _, eles in self.sys.ele_map.items():
            eles._set_external('control_params_src',
                               'broadcast fpdtype_t[2][3]',
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

        if not self.cfg.hasopt(cfgsect, 'mask1') or not self.cfg.hasopt(cfgsect, 'mask2'):
            raise ValueError("[solver-plugin-neuralsource] requires 'mask1' and 'mask2'")

        mask1 = self.cfg.getexpr(cfgsect, 'mask1', subs=subs)
        mask2 = self.cfg.getexpr(cfgsect, 'mask2', subs=subs)

        have_sy = int(self.cfg.hasopt(cfgsect, 'mask1y') and self.cfg.hasopt(cfgsect, 'mask2y'))
        mask1y = self.cfg.getexpr(cfgsect, 'mask1y', subs=subs) if have_sy else '0.0'
        mask2y = self.cfg.getexpr(cfgsect, 'mask2y', subs=subs) if have_sy else '0.0'

        tplargs = dict(
            mask1=mask1,
            mask2=mask2,
            mask1y=mask1y,
            mask2y=mask2y,
            have_sy=have_sy,
            ndims=self.sys.ndims,
            nvars=len(convars),
        )

        # Register the macro on all element types
        for _, eles in self.sys.ele_map.items():
            eles.add_src_macro('pyfr.plugins.kernels.neuralsource',
                               'neuralsource',
                               tplargs,
                               ploc=True,
                               soln=True)

        # Track when the env advances an action
        self._last_step_count = -1

    def __call__(self, intg):
        """Update the ramp endpoints when the env advances to a new action."""
        env = getattr(self.sys, 'env', None)
        if env is None:
            return

        sc = getattr(env, 'step_count', -1)
        if sc != self._last_step_count:
            prev = np.array(env.previous_control[:2], dtype=np.float64)
            curr = np.array(env.current_control[:2], dtype=np.float64)
            t0 = float(env.current_time)

            self.control_params.set(np.array([[prev[0], curr[0], t0],
                                              [prev[1], curr[1], t0]], dtype=np.float64))
            self._last_step_count = sc
