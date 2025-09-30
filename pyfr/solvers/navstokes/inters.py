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

class NavierStokesSubInOutAdiaBCInters(NavierStokesBaseBCInters):
    type = 'sub-inout-adia'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})

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
        self._init_complete = False

    def prepare(self, t):
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            # Direct access to control signal from solver environment
            required_target = self.intg.system.env.current_control[0]
            
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            #print(f"Control signal: {required_target} at time t = {t}") #first setting will be overriden
            self._current_target = required_target
            #print(f"Control signal: {required_target} updated at time t = {t} and step count = {self.intg.system.env.step_count}, last step count = {self.last_step_count}")
            if self._init_complete:
                self.last_step_count = self.intg.system.env.step_count
            else:
                self._init_complete = True

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
        self._init_complete = False

    def prepare(self, t):
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            # Direct access to control signal from solver environment
            required_target = self.intg.system.env.current_control[0]
            
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            self._current_target = required_target
            if self._init_complete:
                self.last_step_count = self.intg.system.env.step_count
            else:
                self._init_complete = True

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
        self._init_complete = False

    def prepare(self, t):
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            # Direct access to control signal from solver environment
            required_target = self.intg.system.env.current_control
            
            self.control_params.set(np.array([[self._current_target, required_target[0], t]]))
            self.control_params2.set(np.array([[self._current_target2,required_target[1]]]))
            self._current_target = required_target[0]
            self._current_target2 = required_target[1]
            if self._init_complete:
                self.last_step_count = self.intg.system.env.step_count
            else:
                self._init_complete = True
class NavierStokesSubInflowFrvNeuralType4BCInters(NavierStokesBaseBCInters): #incomplete
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
        self._init_complete = False

    def prepare(self, t):
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            # Direct access to control signal from solver environment
            required_target = self.intg.system.env.current_control[self.actuator_id]
            
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            #print(f"Control signal: {required_target} at time t = {t}") #first setting will be overriden
            self._current_target = required_target
            #print(f"Control signal: {required_target} updated at time t = {t} and step count = {self.intg.system.env.step_count}, last step count = {self.last_step_count}")
            if self._init_complete:
                self.last_step_count = self.intg.system.env.step_count
            else:
                self._init_complete = True

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
        self._init_complete = False

    def prepare(self, t):
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            # Direct access to control signal from solver environment
            required_target = self.intg.system.env.current_control[self.actuator_id]
            
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            self._current_target = required_target
            if self._init_complete:
                self.last_step_count = self.intg.system.env.step_count
            else:
                self._init_complete = True

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
        self._init_complete = False

    def prepare(self, t):
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            # Direct access to control signal from solver environment
            required_target = -np.sum(self.intg.system.env.current_control)
            
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            self._current_target = required_target
            if self._init_complete:
                self.last_step_count = self.intg.system.env.step_count
            else:
                self._init_complete = True

class NavierStokesSubInflowFrvWsBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv-ws' # weak-specified
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs,
            default={'u': 0, 'v': 0, 'w': 0, 'p': 0}
        )

class NavierStokesSubInflowFpvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-fpv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(
            ['p', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

class NavierStokesAdiaJetBCInters(NavierStokesBaseBCInters):
    type = 'adia-jet'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self.c |= self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
            #['vn'][:1], lhs,
            #default={'vn': 0}
        )

class NavierStokesAdiaJetNeuralType5BCInters(NavierStokesBaseBCInters):
    type = 'adia-jet-neural-type5' # for changing velocity/mass flow rate for multiple actuators
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs,
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
        self._init_complete = False

    def prepare(self, t):
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            # Direct access to control signal from solver environment
            required_target = self.intg.system.env.current_control[self.actuator_id]
            
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            #print(f"Control signal: {required_target} at time t = {t}") #first setting will be overriden
            self._current_target = required_target
            #print(f"Control signal: {required_target} updated at time t = {t} and step count = {self.intg.system.env.step_count}, last step count = {self.last_step_count}")
            if self._init_complete:
                self.last_step_count = self.intg.system.env.step_count
            else:
                self._init_complete = True

class NavierStokesAdiaJetNeuralType5ResidualBCInters(NavierStokesBaseBCInters):
    type = 'adia-jet-neural-type5-residual' # same as type5 but this makes it zero-net-mass-flux
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs,
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
        self._init_complete = False

    def prepare(self, t):
        # Only update backend after environment has taken a step
        if self.intg.system.env.step_count != self.last_step_count:
            # Direct access to control signal from solver environment
            required_target = -np.sum(self.intg.system.env.current_control)
            
            self.control_params.set(np.array([[self._current_target, required_target, t]]))
            self._current_target = required_target
            if self._init_complete:
                self.last_step_count = self.intg.system.env.step_count
            else:
                self._init_complete = True

class NavierStokesAdiaJetNeuralType6BCInters(NavierStokesBaseBCInters):
    """
    Pulsed blowing per jet with 3 RL parameters: amplitude A, duty D, frequency F.
    D and F are held fixed within each action interval. The kernel generates:
        control(t) = 0,                          if t - t0 >= Δt
        control(t) = A * 1[ frac(F * (t - t0)) < D ], otherwise
    Python-side work is minimized: one host->device set per interval with [A,D,F,t0].
    """
    type = 'adia-jet-neural-type6'
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        # Unit jet direction fields from cfg (same pattern as type5)
        self.c |= self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # Action interval Δt: set once
        self.t_act_interval = self.backend.matrix((1, 1))
        self._set_external('t_act_interval',
                           'broadcast fpdtype_t[1][1]',
                           value=self.t_act_interval)
        self.t_act_interval.set(np.array(
            [[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]
        ))

        # Single packed buffer: [A, D, F, t0]
        self.act_pack = self.backend.matrix((1, 4))
        self._set_external('act_pack',
                           'broadcast fpdtype_t[1][4]',
                           value=self.act_pack)

        # Preallocate a small host buffer to avoid reallocations
        self._pack_host = np.zeros((1, 4), dtype=float)
        # Init defaults
        self._pack_host[0, :] = [0.0, 0.10, 10.0, 0.0]
        self.act_pack.set(self._pack_host)

        # Per-jet index and precomputed base offset into the action vector
        self.actuator_id = cfg.getint(cfgsect, 'actuator-number')  # 0-indexed
        self._base = 3 * self.actuator_id

        # Infer bounds once from actions-low/high and store as device constants
        lows = cfg.getliteral('solver-plugin-reinforcementlearning', 'actions-low')
        highs = cfg.getliteral('solver-plugin-reinforcementlearning', 'actions-high')
        try:
            Amin, Dmin, Fmin = float(lows[self._base + 0]), float(lows[self._base + 1]), float(lows[self._base + 2])
            Amax, Dmax, Fmax = float(highs[self._base + 0]), float(highs[self._base + 1]), float(highs[self._base + 2])
        except Exception as e:
            raise ValueError(f"type6 bounds: need 3 entries per jet in actions-low/high; error: {e}")

        # Sanitize for safety; done once
        Dmin = max(0.0, min(1.0, Dmin))
        Dmax = max(0.0, min(1.0, Dmax))
        Fmin = max(0.0, Fmin)

        # Store as compile-time constants for the kernels
        self.c['Amin'], self.c['Amax'] = Amin, Amax
        self.c['Dmin'], self.c['Dmax'] = Dmin, Dmax
        self.c['Fmin'], self.c['Fmax'] = Fmin, Fmax
        self.c['Deps'] = 1.0e-4  # avoid degenerate 0 or 1 duty

        # Minimal Python bookkeeping
        self.last_step_count = -1
        self._init_complete = False

    def prepare(self, t):
        """
        Called very often. Do nothing unless a new RL action arrived.
        On a new action: copy [A, D, F, t] into the preallocated host buffer,
        then one act_pack.set(...) to device. No clipping here; kernel clamps.
        """
        env = self.intg.system.env
        if env.step_count == self.last_step_count:
            return

        # Read the triplet for this jet
        ctrl = env.current_control  # numpy array from your env
        A = float(ctrl[self._base + 0])
        D = float(ctrl[self._base + 1])
        F = float(ctrl[self._base + 2])

        # Pack and push once
        self._pack_host[0, 0] = A
        self._pack_host[0, 1] = D
        self._pack_host[0, 2] = F
        self._pack_host[0, 3] = float(t)
        self.act_pack.set(self._pack_host)

        # Bookkeeping
        if self._init_complete:
            self.last_step_count = env.step_count
        else:
            self._init_complete = True

# -*- coding: utf-8 -*-
import numpy as np
from pyfr.solvers.navstokes.inters import NavierStokesBaseBCInters

class NavierStokesAdiaJetNeuralType7BCInters(NavierStokesBaseBCInters):
    """
    Pulsed blowing with 2 RL parameters per jet: duty D and frequency F.
    Amplitude is fixed by the cfg 'u','v','w' expressions (slot profile),
    and the kernel applies a square gate g(t) with duty/frequency.

        control(t) = 0,                           if t - t0 >= Δt
        control(t) = 1[ frac(F * (t - t0)) < D ], otherwise

    Only one host->device set per action interval with [D, F, t0].
    """
    type = 'adia-jet-neural-type7'
    cflux_state = 'ghost'

    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        self.intg = intg
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        # Fixed jet profile fields from cfg (same pattern as type 5/6)
        # These can vary with x,y and already encode the amplitude and direction.
        self.c |= self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # Action interval Δt: set once (shared for all actions)
        self.t_act_interval = self.backend.matrix((1, 1))
        self._set_external('t_act_interval',
                           'broadcast fpdtype_t[1][1]',
                           value=self.t_act_interval)
        self.t_act_interval.set(np.array(
            [[cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')]]
        ))

        # Single packed buffer: [D, F, t0]
        self.act_pack = self.backend.matrix((1, 3))
        self._set_external('act_pack',
                           'broadcast fpdtype_t[1][3]',
                           value=self.act_pack)

        # Preallocate a small host buffer to avoid reallocations
        self._pack_host = np.zeros((1, 3), dtype=float)
        # Init defaults: D in (0,1), F >= 0
        self._pack_host[0, :] = [0.10, 10.0, 0.0]
        self.act_pack.set(self._pack_host)

        # Per-jet index and precomputed base offset into the action vector
        self.actuator_id = cfg.getint(cfgsect, 'actuator-number')  # 0-indexed
        self._base = 2 * self.actuator_id  # 2 params per jet: D, F

        # Infer bounds once from actions-low/high and store as device constants
        lows = cfg.getliteral('solver-plugin-reinforcementlearning', 'actions-low')
        highs = cfg.getliteral('solver-plugin-reinforcementlearning', 'actions-high')
        try:
            Dmin = float(lows[self._base + 0]); Fmin = float(lows[self._base + 1])
            Dmax = float(highs[self._base + 0]); Fmax = float(highs[self._base + 1])
        except Exception as e:
            raise ValueError(f"type7 bounds: need 2 entries per jet in actions-low/high; error: {e}")

        # Sanitize for safety; done once
        Dmin = max(0.0, min(1.0, Dmin))
        Dmax = max(0.0, min(1.0, Dmax))
        Fmin = max(0.0, Fmin)

        # Store as compile-time constants for the kernels
        self.c['Dmin'], self.c['Dmax'] = Dmin, Dmax
        self.c['Fmin'], self.c['Fmax'] = Fmin, Fmax
        self.c['Deps'] = 1.0e-4  # avoid degenerate 0 or 1 duty

        # Minimal Python bookkeeping
        self.last_step_count = -1
        self._init_complete = False

    def prepare(self, t):
        """
        Called often. Only act when a new RL action arrives.
        On a new action: copy [D, F, t] into the preallocated host buffer,
        then one act_pack.set(...) to device. No clipping here; kernel clamps.
        """
        env = self.intg.system.env
        if env.step_count == self.last_step_count:
            return

        # Read the pair for this jet
        ctrl = env.current_control  # numpy array from your env
        D = float(ctrl[self._base + 0])
        F = float(ctrl[self._base + 1])

        # Pack and push once
        self._pack_host[0, 0] = D
        self._pack_host[0, 1] = F
        self._pack_host[0, 2] = float(t)
        self.act_pack.set(self._pack_host)

        # Bookkeeping
        if self._init_complete:
            self.last_step_count = env.step_count
        else:
            self._init_complete = True
