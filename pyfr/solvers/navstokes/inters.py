import numpy as np
from rtree.index import Index, Property

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, init_csv
from pyfr.quadrules import get_quadrule

from pyfr.plugins.base import BaseSolverPlugin

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)
from pyfr.solvers.euler.inters import (FluidIntIntersMixin,
                                       FluidMPIIntersMixin)


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = self.cfg.get('solver', 'shock-capturing')
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
                            FluidIntIntersMixin,
                            BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs,
            ulout=self._comm_lhs, urout=self._comm_rhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            nl=self._pnorm_lhs
        )


class NavierStokesMPIInters(TplargsMixin,
                            FluidMPIIntersMixin,
                            BaseAdvectionDiffusionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs, ulout=self._comm_lhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            nl=self._pnorm_lhs
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
            extrns=self._external_args, ulin=self._scal_lhs,
            ulout=self._comm_lhs, nlin=self._pnorm_lhs,
            **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs,
            artviscl=self._artvisc_lhs, **self._external_vals
        )

        if self._ef_enabled:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.bccent'
            )

            self.kernels['comm_entropy'] = lambda: self._be.kernel(
                'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
                extrns=self._external_args, entmin_lhs=self._entmin_lhs,
                nl=self._pnorm_lhs, ul=self._scal_lhs, **self._external_vals
            )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost'


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

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

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(['p'], lhs)

class NavierStokesSubInflowFrvNeuralBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv-neural' # for changing velocity/mass flow rate
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # some config parameters
        self.t_act_interval = self.backend.matrix((1,1))
        self._set_external('t_act_interval', 'broadcast fpdtype_t[1][1]', 
                         value=self.t_act_interval)

        # Neural network + control parameters
        self.control_params = self.backend.matrix((1,3))
        self._set_external('control_params', 'broadcast fpdtype_t[1][3]', 
                         value=self.control_params)

        # Initial value
        self.control_params.set(np.array([[0.0, 0.0, 0.0]])) #(Q0,Q1,t0)

        # Cache current parameter value 
        self._current_target = 0.0

        # Fixed values
        self.t_act_interval.set(np.array([[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]))
        #print(f"jcenter: {jcenter[0][0]}, polarity: {polarity[0][0]}")

        # Helper to keep track of last step count
        self.last_step_count = -1

    def prepare(self, t):
        # Direct access to control signal from solver environment
        new_targets = self.intg.system.env.current_control

        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            self.control_params.set(np.array([[self._current_target, new_targets[0], t]]))
            #print(f"Control signal: {new_targets[0]} at time t = {t}")
            self._current_target = new_targets[0]
            #print(f"Control signal: {new_targets[0]} updated at time t = {t} and step count = {self.intg.system.env.step_count}, last step count = {self.last_step_count}")
            self.last_step_count = self.intg.system.env.step_count

class NavierStokesSubInflowFrvNeuralType2BCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv-neural-type2' # for changing angles
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # some config parameters
        self.t_act_interval = self.backend.matrix((1,1))
        self._set_external('t_act_interval', 'broadcast fpdtype_t[1][1]', 
                         value=self.t_act_interval)

        # Neural network + control parameters
        self.control_params = self.backend.matrix((1,3))
        self._set_external('control_params', 'broadcast fpdtype_t[1][3]', 
                         value=self.control_params)

        # Initial value
        self.control_params.set(np.array([[0.0, 0.0, 0.0]])) #(Q0,Q1,t0)

        # Cache current parameter value 
        self._current_target = 0.0

        # Fixed values
        self.t_act_interval.set(np.array([[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]))

        self.last_step_count = -1

    def prepare(self, t):
        # Direct access to control signal from solver environment
        new_targets = self.intg.system.env.current_control

        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            self.control_params.set(np.array([[self._current_target, new_targets[0], t]]))
            self._current_target = new_targets[0]
            self.last_step_count = self.intg.system.env.step_count

class NavierStokesSubInflowFrvNeuralType3BCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv-neural-type3' # for changing both velocity/mfr and angles
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # some config parameters
        self.t_act_interval = self.backend.matrix((1,1))
        self._set_external('t_act_interval', 'broadcast fpdtype_t[1][1]', 
                         value=self.t_act_interval)

        # Neural network + control parameters
        self.control_params = self.backend.matrix((1,3))
        self._set_external('control_params', 'broadcast fpdtype_t[1][3]', 
                         value=self.control_params)
        self.control_params2 = self.backend.matrix((1,2))
        self._set_external('control_params2', 'broadcast fpdtype_t[1][2]', 
                         value=self.control_params2)

        # Initial value
        self.control_params.set(np.array([[0.0, 0.0, 0.0]])) #(Q0,Q1,t0)
        self.control_params2.set(np.array([[0.0, 0.0]])) #(Q0,Q1) for 2nd control parameter

        # Cache current parameter value 
        self._current_target = 0.0
        self._current_target2 = 0.0

        # Fixed values
        self.t_act_interval.set(np.array([[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]))

        self.last_step_count = -1

    def prepare(self, t):
        # Direct access to control signal from solver environment
        new_targets = self.intg.system.env.current_control

        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            self.control_params.set(np.array([[self._current_target, new_targets[0], t]]))
            self.control_params2.set(np.array([[self._current_target2, new_targets[1]]]))
            self._current_target = new_targets[0]
            self._current_target2 = new_targets[1]
            self.last_step_count = self.intg.system.env.step_count

class NavierStokesSubInflowFrvNeuralType4BCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv-neural-type4' # for changing velocity/mass flow rate; discrete action space
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # some config parameters
        self.t_act_interval = self.backend.matrix((1,1))
        self._set_external('t_act_interval', 'broadcast fpdtype_t[1][1]', 
                         value=self.t_act_interval)

        # Neural network + control parameters
        self.control_params = self.backend.matrix((1,3))
        self._set_external('control_params', 'broadcast fpdtype_t[1][3]', 
                         value=self.control_params)

        # Initial value
        self.control_params.set(np.array([[0.0, 0.0, 0.0]])) #(Q0,Q1,t0)

        # Cache current parameter value 
        self._current_target = 0.0

        # Fixed values
        self.t_act_interval.set(np.array([[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]))
        self.actuator_id = cfg.getint(cfgsect, 'actuator-number')
        #print(f"Actuator number: {self.actuator_id}")
        if self.actuator_id < 1:
            raise ValueError("actuator-number must be >= 1")

        # Get and validate action range (here actions are just indices of possible configurations, not actions themselves)
        try:
            self.actions_low = int(self.cfg.getliteral('solver-plugin-reinforcementlearning', 'actions-low')[0])
            self.actions_high = int(self.cfg.getliteral('solver-plugin-reinforcementlearning', 'actions-high')[0])
            
            if not isinstance(self.actions_low, int) or not isinstance(self.actions_high, int):
                raise ValueError("Action bounds must be integers")
                
            if self.actions_low >= self.actions_high:
                raise ValueError("actions-low must be less than actions-high")
                
            self.action_space_size = self.actions_high - self.actions_low + 1
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid action bounds: {e}")

        #active_configs = self._find_active_configs()
        #print(f"\nActuator {self.actuator_id} configuration summary:")
        #print(f"Will turn ON for configurations: {active_configs}")
        #print(f"Binary patterns that activate this actuator:")
        #for config in active_configs:
        #    binary = format(config, f'0{int(np.log2(self.action_space_size))}b')
        #    print(f"Config {config}: {binary}")

        # Helper to keep track of last step count
        self.last_step_count = -1

    def _get_binary_config(self, one_hot_action):
        """Convert one-hot action to binary configuration."""
        if one_hot_action is None or len(one_hot_action) == 0:
            # Return default configuration (all zeros)
            return [0] * int(np.log2(self.action_space_size))
            
        try:
            config_idx = np.where(one_hot_action == 1)[0][0]
            binary = format(config_idx, f'0{int(np.log2(self.action_space_size))}b')
            return [int(bit) for bit in binary]
        except (IndexError, AttributeError):
            # Return default configuration if something goes wrong
            return [0] * int(np.log2(self.action_space_size))

    def _find_active_configs(self):
        """Find configurations where this actuator is active."""
        active_configs = []
        num_bits = int(np.log2(self.action_space_size))
        
        for config_idx in range(self.action_space_size):
            # Convert to binary string with proper padding
            binary = format(config_idx, f'0{num_bits}b')
            # Check if bit at actuator position is 1 (right to left, 1-based)
            if binary[-self.actuator_id] == '1':
                active_configs.append(config_idx)
        return active_configs

    def prepare(self, t):
        new_targets = getattr(self.intg.system.env, 'current_control', None)
        
        if self.intg.system.env.step_count != self.last_step_count:
            binary_config = self._get_binary_config(new_targets)
            is_active = binary_config[-self.actuator_id]
            Q1 = float(is_active)
            #print(f"Actuator {self.actuator_id} configuration: {binary_config}, active: {is_active}, Q: {Q1}")
            
            self.control_params.set(np.array([[self._current_target, Q1, t]]))
            self._current_target = Q1
            self.last_step_count = self.intg.system.env.step_count

class NavierStokesSubInflowFrvNeuralType5BCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv-neural-type5' # for changing velocity/mass flow rate for multiple actuators
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # some config parameters
        self.t_act_interval = self.backend.matrix((1,1))
        self._set_external('t_act_interval', 'broadcast fpdtype_t[1][1]', 
                         value=self.t_act_interval)

        # Neural network + control parameters
        self.control_params = self.backend.matrix((1,3))
        self._set_external('control_params', 'broadcast fpdtype_t[1][3]', 
                         value=self.control_params)

        # Initial value
        self.control_params.set(np.array([[0.0, 0.0, 0.0]])) #(Q0,Q1,t0)

        # Cache current parameter value 
        self._current_target = 0.0

        # Fixed values
        self.t_act_interval.set(np.array([[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]))
        self.actuator_id = cfg.getint(cfgsect, 'actuator-number') # 0-indexed

        # Helper to keep track of last step count
        self.last_step_count = -1

    def prepare(self, t):
        # Direct access to control signal from solver environment
        new_targets = self.intg.system.env.current_control
        required_target = new_targets[self.actuator_id]

        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            #print(f"Control signal: {required_target} at time t = {t}")
            self._current_target = required_target
            #print(f"Control signal: {required_target} updated at time t = {t} and step count = {self.intg.system.env.step_count}, last step count = {self.last_step_count}")
            self.last_step_count = self.intg.system.env.step_count

class NavierStokesCharRiemInvNeuralBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv-neural' # for changing velocity/mass flow rate
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization : careful, don't make rho or p a function of uvw
        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )

        # some config parameters
        self.t_act_interval = self.backend.matrix((1,1))
        self._set_external('t_act_interval', 'broadcast fpdtype_t[1][1]', 
                         value=self.t_act_interval)

        # Neural network + control parameters
        self.control_params = self.backend.matrix((1,3))
        self._set_external('control_params', 'broadcast fpdtype_t[1][3]', 
                         value=self.control_params)

        # Initial value
        self.control_params.set(np.array([[0.0, 0.0, 0.0]])) #(Q0,Q1,t0)

        # Cache current parameter value 
        self._current_target = 0.0

        # Fixed values
        self.t_act_interval.set(np.array([[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]))

        # Helper to keep track of last step count
        self.last_step_count = -1

    def prepare(self, t):
        # Direct access to control signal from solver environment
        new_targets = self.intg.system.env.current_control
        #print("stepping")
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            self.control_params.set(np.array([[self._current_target, new_targets[0], t]]))
            self._current_target = new_targets[0]
            #print(f"Control signal: {new_targets[0]} at time t = {t}")
            self.last_step_count = self.intg.system.env.step_count

class NavierStokesCharRiemInvNeuralType5BCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv-neural-type5' # for changing velocity or mass flow rate for multiple actuators
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization : careful, don't make rho or p a function of uvw
        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )

        # some config parameters
        self.t_act_interval = self.backend.matrix((1,1))
        self._set_external('t_act_interval', 'broadcast fpdtype_t[1][1]', 
                         value=self.t_act_interval)

        # Neural network + control parameters
        self.control_params = self.backend.matrix((1,3))
        self._set_external('control_params', 'broadcast fpdtype_t[1][3]', 
                         value=self.control_params)

        # Initial value
        self.control_params.set(np.array([[0.0, 0.0, 0.0]])) #(Q0,Q1,t0)

        # Cache current parameter value 
        self._current_target = 0.0

        # Fixed values
        self.t_act_interval.set(np.array([[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]))
        self.actuator_id = cfg.getint(cfgsect, 'actuator-number') # 0-indexed

        # Helper to keep track of last step count
        self.last_step_count = -1

    def prepare(self, t):
        # Direct access to control signal from solver environment
        new_targets = self.intg.system.env.current_control
        required_target = new_targets[self.actuator_id]

        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            self._current_target = required_target
            self.last_step_count = self.intg.system.env.step_count

class NavierStokesSubInflowFrvNeuralType5ResidualBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv-neural-type5-residual' # same as type5 but this makes it zero-net-mass-flux
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # some config parameters
        self.t_act_interval = self.backend.matrix((1,1))
        self._set_external('t_act_interval', 'broadcast fpdtype_t[1][1]', 
                         value=self.t_act_interval)

        # Neural network + control parameters
        self.control_params = self.backend.matrix((1,3))
        self._set_external('control_params', 'broadcast fpdtype_t[1][3]', 
                         value=self.control_params)

        # Initial value
        self.control_params.set(np.array([[0.0, 0.0, 0.0]])) #(Q0,Q1,t0)

        # Cache current parameter value 
        self._current_target = 0.0

        # Fixed values
        self.t_act_interval.set(np.array([[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]))

        # Helper to keep track of last step count
        self.last_step_count = -1

    def prepare(self, t):
        new_targets = self.intg.system.env.current_control
        required_target = -np.sum(new_targets) # zero-net-mass-flux

        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            self._current_target = required_target
            self.last_step_count = self.intg.system.env.step_count