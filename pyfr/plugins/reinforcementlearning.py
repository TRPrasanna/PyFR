from pyfr.plugins.base import BaseSolverPlugin
from pyfr.quadrules import get_quadrule
from pyfr.plugins.sampler import _closest_pts, _plocs_to_tlocs
from pyfr.mpiutil import get_comm_rank_root, mpi
import numpy as np
import torch
import ast
import operator
import math
from collections import defaultdict
from pyfr.plugins.base import BaseSolnPlugin, SurfaceMixin
from scipy.integrate import trapezoid

class ReinforcementLearningPlugin(BaseSolverPlugin, SurfaceMixin, BaseSolnPlugin):
    name = 'reinforcementlearning'
    systems = ['ac-navier-stokes', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [2, 3]
    
    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()
        self.device = torch.device('cpu') # check: find a way to use value from config
        # Get sampling points configuration
        self.pts = self.cfg.getliteral(cfgsect, 'probe-pts')
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')
        
        # Setup sampling infrastructure
        self._setup_sampling(intg)
        
        # Calculate observation size based on probe points and variables
        default_var_list = ['u', 'v', 'p']
        var_string = self.cfg.get(cfgsect, 'observation-variables',
                                  ','.join(default_var_list))
        self.obs_var_names = [v.strip() for v in var_string.replace(',', ' ').split()]
        primitive_names = list(self.elementscls.privarmap[self.ndims]) # e.g. 2-D: ['rho', 'u', 'v', 'p'] (compressible, check for inc.)
        try:
            self.var_indices = [primitive_names.index(v) for v in self.obs_var_names]
        except ValueError as err:
            raise ValueError(f"[reinforcementlearning] observation-variables: "
                             f"unknown name in {self.obs_var_names}; "
                             f"valid choices: {primitive_names}") from err
        base_obs = len(self.pts) * len(self.var_indices)

        # Environment reference for goal-conditioned RL (if available)
        self.env = getattr(intg.system, 'env', None)
        self.goal_dim = getattr(self.env, 'num_reward_targets', 0) if self.env else 0
        self.observation_size = base_obs + (self.goal_dim or 0)
        self.obs_var_names = [v.strip() for v in var_string.replace(',', ' ').split()] # to print for diagnostics
        #nvars = len(self.elementscls.privarmap[self.ndims]) if self.fmt == 'primitive' else len(self.elementscls.convarmap[self.ndims])
        #self.observation_size = len(self.pts) #* 2 #* 3 # * nvars for all variables
        #self.nvars = nvars
        
        # Rest of initialization
        self.action_interval = self.cfg.getfloat(cfgsect, 'action-interval', 0.1)
        self.last_action_time = intg.tcurr

        # Force and moments calculation setup (from FluidForcePlugin)
        self._viscous = 'navier-stokes' in intg.system.name
        self._ac = intg.system.name.startswith('ac')
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')
        self._constants = self.cfg.items_as('constants', float)

        # Moments : check how to avoid calculating Cl and Cd when only this is required
        mcomp = 3 if self.ndims == 3 else 1
        self._mcomp = mcomp if self.cfg.hasopt(cfgsect, 'morigin') else 0
        if self._mcomp:
            self.morigin = morigin = np.array(self.cfg.getliteral(cfgsect, 'morigin'))
            if len(morigin) != self.ndims:
                raise ValueError(f'morigin must have {self.ndims} components')

        # Read multiple surface names
        self.surf_bnames = self.cfg.getliteral(cfgsect, 'surfaces')
        if not isinstance(self.surf_bnames, list):
            self.surf_bnames = [self.surf_bnames]
        if not self.surf_bnames:
            raise ValueError("No surfaces specified for forces/moment calculation")

        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # Store matrices for each surface
        self._m0 = defaultdict(dict)
        self._qwts = defaultdict(lambda: defaultdict(list))
        self._eidxs = defaultdict(dict)
        self._norms = defaultdict(dict)
        self._rfpts = defaultdict(dict) if self._mcomp else None

        if self._viscous:
            self._m4 = defaultdict(dict)
            self._rcpjact = defaultdict(dict)

        # Check each boundary's existence across ranks
        self.rallocs = intg.rallocs # save for compute_fm
        for surf in self.surf_bnames:
            bc = f'bcon_{surf}_p{intg.rallocs.prank}'
            bcranks = comm.gather(bc in mesh, root=root)

            # Exit if boundary not found
            if rank == root:
                if not any(bcranks):
                    raise RuntimeError(f'Boundary {surf} does not exist')

            # Initialize matrices if boundary exists in this rank
            if bc in intg.system.mesh:
                self._init_surface(intg, bc, surf)

        # Add step-based sampling (like SamplerPlugin)
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps', 10)
        #print(f"Sampling forces every {self.nsteps} steps")
            
        # Initialize force history buffers
        self.force_times = []
        self.drag_history = []
        self.lift_history = []
        self.moment_history = []
        self.action_history = []
        self.avg_window = self.cfg.getfloat(cfgsect, 'averaging-window', 0.5)
        
        # Read reward function configuration
        self.reward_function = self.cfg.get(cfgsect, 'reward-function', 
                                           '-abs(avg_moment)')  # Default reward
        
        # Configure statistics options
        self.normalize_reward = self.cfg.getbool(cfgsect, 'normalize-reward', False)
        
        # Reference values for normalization (optional) - now from main section
        self.ref_values = {}
        if self.normalize_reward:
            # Look for reference values directly in the main section
            for key in ['drag-ref', 'lift-ref', 'moment-ref', 'action-ref']:
                if self.cfg.hasopt(cfgsect, key):
                    # Store with simplified keys (without the -ref suffix)
                    simple_key = key.replace('-ref', '')
                    self.ref_values[simple_key] = self.cfg.getfloat(cfgsect, key)
        
        # Initialize expression evaluator
        self.expr_evaluator = SafeExpressionEvaluator()
        
        # Determine which variables are used in the reward function
        self.used_variables = self.expr_evaluator.find_used_variables(self.reward_function)
        
        # Print summary of which metrics will be calculated
        #print(f"Reward function: {self.reward_function}")
        #print(f"Variables used in reward function: {', '.join(sorted(self.used_variables))}")

    def _init_surface(self, intg, bc, surf):
        """Initialize matrices for a single surface"""
        mesh, elemap = intg.system.mesh, intg.system.ele_map
        
        # Grab sub-dictionaries for this surface instead of overwriting
        m0 = self._m0[surf]
        qwts = self._qwts[surf]
        eidxs = self._eidxs[surf]
        norms = self._norms[surf]
        
        if self._mcomp:
            rfpts = self._rfpts[surf]
        
        if self._viscous:
            m4 = self._m4[surf]
            rcpjact = self._rcpjact[surf]

        for etype, eidx_, fidx, flags in mesh[bc].tolist():
            eles = elemap[etype]
            itype, proj, norm = eles.basis.faces[fidx]
            
            ppts, pwts = self._surf_quad(itype, proj, flags='s')
            pnorm = eles.pnorm_at(ppts, [norm]*len(ppts))[:, eidx_]
            
            key = (etype, fidx)
            eidxs.setdefault(key, []).append(eidx_)
            norms.setdefault(key, []).append(pnorm)

            if key not in m0:
                m0[key] = eles.basis.ubasis.nodal_basis_at(ppts)
                qwts[key] = pwts

                if self._viscous and etype not in m4:
                    m4[etype] = eles.basis.m4
                    smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
                    djac = eles.rcpdjac_at_np('upts')
                    rcpjact[etype] = smat * djac

            if self._mcomp:
                ploc = eles.ploc_at_np(ppts)[..., eidx_]
                rfpts.setdefault(key, []).append(ploc - self.morigin)

        # Convert lists to arrays
        self._eidxs[surf] = {k: np.array(v) for k, v in eidxs.items()}
        self._norms[surf] = {k: np.array(v) for k, v in norms.items()}
        if self._mcomp:
            self._rfpts[surf] = {k: np.array(v) for k, v in rfpts.items()}
        if self._viscous:
            self._rcpjact[surf] = {
                k: rcpjact[k[0]][..., self._eidxs[surf][k]] for k in self._eidxs[surf]
            }

    def __call__(self, intg):
        """Called after each step - store forces/moments if needed""" # __call__ in every plugin is called every time step
        # Return if no sampling is due
        if intg.nacptsteps % self.nsteps:
            return

        previous_control_target = intg.system.env.previous_control
        current_control_target = intg.system.env.current_control
        #current_control_value = (intg.system.env.current_control-intg.system.env.previous_control)/intg.system.env.action_interval*(intg.tcurr-intg.system.env.current_time) + intg.system.env.previous_control
        # possibly can minmax time instead
        current_control_value = (current_control_target-previous_control_target)/intg.system.env.action_interval*(intg.tcurr-intg.system.env.current_time) + previous_control_target
        lower_bound = np.minimum(previous_control_target,current_control_target)
        upper_bound = np.maximum(previous_control_target,current_control_target)
        current_control_value = np.maximum(lower_bound, np.minimum(current_control_value, upper_bound))
        #Q = (Q1-Q0)/Ta * (t-t0) + Q0; but this ramping behaviour may change in future; check
        #print(f"Current control value: {current_control_value}", previous_control_target, current_control_target)

        # store sum of absolute values of actions
        #self.sumabsact = np.sum(np.abs(current_control_value))+abs(np.sum(current_control_value)) # DRL jets + opposing ZNMF jet
        self.sumabsact = np.sum(current_control_value**2)+np.sum(current_control_value)**2 # not really sub of absolute
        #print(f"Sum of absolute actions: {self.sumabsact}")

        # Get forces and store them
        comm, rank, root = get_comm_rank_root()
        fm = self._compute_fm(intg, dict(zip(intg.system.ele_types, intg.soln)))
        #print(fm) # check how to throw error if morigin is not given in .ini file
        if rank == root:
            t = intg.tcurr
            drag = (fm[0, 0] + (fm[1, 0] if self._viscous else 0) + fm[2, 0]) * 2
            lift = (fm[0, 1] + (fm[1, 1] if self._viscous else 0) + fm[2, 1]) * 2
            if self._mcomp:
                moment = (fm[0, 2] + (fm[1, 2] if self._viscous else 0) + fm[2, 2]) * 2
            # needs to be reconfigured for 3D cases! check;
            
            # Store forces
            self.force_times.append(t)
            self.drag_history.append(drag)
            self.lift_history.append(lift)
            if self._mcomp:
                self.moment_history.append(moment)
            # store sum(|actions|)
            self.action_history.append(self.sumabsact)
            
            # Remove old data outside window
            while self.force_times[0] < t - self.avg_window:
                self.force_times.pop(0)
                self.drag_history.pop(0)
                self.lift_history.pop(0)
                self.action_history.pop(0)
                if self._mcomp:
                    self.moment_history.pop(0)

    def _compute_fm(self, intg, solns):
        """Compute instantaneous forces/moments for all surfaces"""
        comm, rank, root = get_comm_rank_root()
        
        # Initialize arrays
        ndims = self.ndims
        mcomp = self._mcomp
        # Same array structure as original
        fm = np.zeros((3 if self._viscous else 2, ndims + mcomp))
        
        # Process each surface, accumulating forces in fm
        for surf in self.surf_bnames:
            bc = f'bcon_{surf}_p{intg.rallocs.prank}'
            if bc not in intg.system.mesh:
                continue
                
            # Process each element type, following original logic
            for etype, fidx in self._m0[surf]:
                # Get interpolation operator
                m0 = self._m0[surf][etype, fidx]
                nfpts, nupts = m0.shape
                
                # Get solution at points
                uupts = solns[etype][..., self._eidxs[surf][etype, fidx]]
                
                # Interpolate to face
                ufpts = m0 @ uupts.reshape(nupts, -1)
                ufpts = ufpts.reshape(nfpts, self.nvars, -1)
                ufpts = ufpts.swapaxes(0, 1)
                
                # Compute pressure
                pidx = 0 if self._ac else -1
                p = self.elementscls.con_to_pri(ufpts, self.cfg)[pidx]
                
                # Get weights and normals
                qwts = self._qwts[surf][etype, fidx]
                norms = self._norms[surf][etype, fidx]
                
                # Pressure force
                fm[0, :ndims] += np.einsum('i...,ij,jik', qwts, p, norms)
                
                # Force from momentum flux (same as original)
                pri_vars = self.elementscls.con_to_pri(ufpts, self.cfg)
                vs = np.array(pri_vars[1:-1])
                rho = np.ones_like(vs[0]) if self._ac else pri_vars[0]
                rhovs = rho[None, :, :] * vs
                fm[2, :ndims] += np.einsum('i,jim,mij,kim->k', qwts, rhovs, norms, vs)
                
                if self._viscous:
                    # Get viscous terms (same as original)
                    m4 = self._m4[surf][etype]
                    rcpjact = self._rcpjact[surf][etype, fidx]
                    
                    # Transformed gradient at solution points
                    tduupts = m4 @ uupts.reshape(nupts, -1)
                    tduupts = tduupts.reshape(ndims, nupts, self.nvars, -1)
                    
                    # Physical gradient at solution points
                    duupts = np.einsum('ijkl,jkml->ikml', rcpjact, tduupts)
                    duupts = duupts.reshape(ndims, nupts, -1)
                    
                    # Interpolate gradient to flux points
                    dufpts = np.array([m0 @ du for du in duupts])
                    dufpts = dufpts.reshape(ndims, nfpts, self.nvars, -1)
                    dufpts = dufpts.swapaxes(1, 2)
                    
                    # Viscous stress
                    if self._ac:
                        vis = self.ac_stress_tensor(dufpts)
                    else:
                        vis = self.stress_tensor(ufpts, dufpts)
                    
                    # Add to forces
                    fm[1, :ndims] += np.einsum('i...,klij,jil', qwts, vis, norms)
                
                if self._mcomp:
                    # Moment calculations (same as original)
                    rfpts = self._rfpts[surf][etype, fidx]
                    rcn = np.atleast_3d(np.cross(rfpts, norms))
                    
                    # Pressure moments
                    fm[0, ndims:] += np.einsum('i...,ij,jik->k', qwts, p, rcn)
                    
                    # Momentum flux moments
                    momflux = np.einsum('jim,mij,kim->kim', rhovs, norms, vs)
                    rcf = np.atleast_3d(np.cross(rfpts, momflux.T))
                    fm[2, ndims:] += np.einsum('i,jik->k', qwts, rcf)
                    
                    if self._viscous:
                        # Viscous moments
                        viscf = np.einsum('ijkl,lkj->lki', vis, norms)
                        rcf = np.atleast_3d(np.cross(rfpts, viscf))
                        fm[1, ndims:] += np.einsum('i,jik->k', qwts, rcf)
        
        # Reduce across ranks (same as original)
        if rank != root:
            comm.Reduce(fm, None, op=mpi.SUM, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, fm, op=mpi.SUM, root=root)
        
        return fm

    def _setup_sampling(self, intg):
        """Setup sampling infrastructure similar to SamplerPlugin"""
        # Store elementscls for variable mapping
        self.elementscls = intg.system.elementscls
        
        # Get MPI info
        comm, rank, root = get_comm_rank_root()
        
        # Find search points in physical and transformed space
        self.tlocs, self.plocs = self._search_pts(intg)
        
        # Find closest points and refine
        closest = _closest_pts(self.plocs, self.pts)
        elepts = [[] for i in range(len(intg.system.ele_map))]
        
        for i, (dist, etype, (uidx, eidx)) in enumerate(closest):
            # Find closest point across ranks
            _, mrank = comm.allreduce((dist, rank), op=mpi.MINLOC)
            if rank == mrank:
                elepts[etype].append((i, eidx, self.tlocs[etype][uidx]))
                
        # Refine points
        self._ourpts = self._refine_pts(intg, elepts)

    def _search_pts(self, intg):
        """Find sampling points in physical and transformed space"""
        tlocs, plocs = [], []
        
        # Use quadrature points as search locations
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn', 
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }
        
        for etype, eles in intg.system.ele_map.items():
            pts = get_quadrule(etype, qrule_map[etype], eles.basis.nupts).pts
            tlocs.append(pts)
            plocs.append(eles.ploc_at_np(pts).swapaxes(1, 2))
            
        return tlocs, plocs

    def _refine_pts(self, intg, elepts):
        elelist = intg.system.ele_map.values()
        ptsinfo = []

        # Loop over all the points for each element type
        for etype, (eles, epts) in enumerate(zip(elelist, elepts)):
            if not epts:
                continue

            idx, eidx, tlocs = zip(*epts)
            spts = eles.eles[:, eidx, :]
            plocs = [self.pts[i] for i in idx]

            # Use Newton's method to find the precise transformed locations
            ntlocs, nplocs = _plocs_to_tlocs(eles.basis.sbasis, spts, plocs,
                                             tlocs)

            # Form the corresponding interpolation operators
            intops = eles.basis.ubasis.nodal_basis_at(ntlocs)

            # Append to the point info list
            ptsinfo.extend(
                (*info, etype) for info in zip(idx, eidx, nplocs, intops)
            )

        # Sort our info array by its original index
        ptsinfo.sort()

        # Strip the index, move etype to the front, and return
        return [(etype, *info) for idx, *info, etype in ptsinfo]
    
    def _get_observation(self, solver):
        """Get flow values at probe points"""
        # Get solution matrices
        solns = solver.soln
        
        # Sample and interpolate at probe points
        samples = [op @ solns[et][:, :, ei] for et, ei, _, op in self._ourpts]
        
        # Convert to primitive variables if needed
        if self.fmt == 'primitive' and samples:
            samples = self.elementscls.con_to_pri(np.array(samples).T, self.cfg)
            samples = np.array(samples).T

            # Extract rho, u, v, p from samples depending on requested variables
            samples = samples[:, self.var_indices]

        # Convert to tensor of 32-bit floats, check
        #print(f"Samples: {samples}")
        obs = torch.tensor(samples, device=self.device).flatten().float()

        # Append goal targets if enabled
        if self.goal_dim and self.env is not None:
            tgt_tensor = self.env.get_current_targets_tensor()
            if tgt_tensor is not None:
                obs = torch.cat((obs, tgt_tensor.to(self.device)))

        return obs

    def _compute_std(self, history, mean):
        """Compute time-weighted standard deviation of a signal"""
        if len(history) <= 1 or len(self.force_times) <= 1:
            return 0.0
            
        # Calculate squared differences from mean
        sq_diffs = [(x - mean)**2 for x in history]
        
        # Integrate squared differences over time
        variance = trapezoid(y=sq_diffs, x=self.force_times) / (self.force_times[-1] - self.force_times[0])
        
        # Return standard deviation
        return np.sqrt(variance)

    def _get_reward(self, solver):
        """Compute reward using stored force history and configured reward function"""
        if not self.force_times:
            return 0.0  # No data yet
            
        variables = {}  # Dictionary to hold only needed variables
        
        # Only calculate time-averaged values if needed
        needs_avg_drag = 'avg_drag' in self.used_variables
        needs_avg_lift = 'avg_lift' in self.used_variables
        needs_avg_moment = 'avg_moment' in self.used_variables
        needs_avg_sumabsact = 'avg_sumabsact' in self.used_variables
        
        # Standard deviations
        needs_std_drag = 'std_drag' in self.used_variables
        needs_std_lift = 'std_lift' in self.used_variables
        needs_std_moment = 'std_moment' in self.used_variables
        
        # Get time window for integration
        delta_t = self.force_times[-1] - self.force_times[0]
        if delta_t <= 0:
            return 0.0  # Avoid division by zero
        
        # Calculate only the requested averages
        if needs_avg_drag:
            variables['avg_drag'] = trapezoid(y=self.drag_history, x=self.force_times) / delta_t
            
        if needs_avg_lift:
            variables['avg_lift'] = trapezoid(y=self.lift_history, x=self.force_times) / delta_t
            
        if needs_avg_moment and self._mcomp and self.moment_history:
            variables['avg_moment'] = trapezoid(y=self.moment_history, x=self.force_times) / delta_t
        elif needs_avg_moment:
            variables['avg_moment'] = 0.0
            
        if needs_avg_sumabsact:
            variables['avg_sumabsact'] = trapezoid(y=self.action_history, x=self.force_times) / delta_t
        
        # Calculate only the requested standard deviations
        if needs_std_drag:
            if not needs_avg_drag:  # Calculate avg_drag if not already done
                avg_drag = trapezoid(y=self.drag_history, x=self.force_times) / delta_t
            else:
                avg_drag = variables['avg_drag']
            variables['std_drag'] = self._compute_std(self.drag_history, avg_drag)
            
        if needs_std_lift:
            if not needs_avg_lift:  # Calculate avg_lift if not already done
                avg_lift = trapezoid(y=self.lift_history, x=self.force_times) / delta_t
            else:
                avg_lift = variables['avg_lift']
            variables['std_lift'] = self._compute_std(self.lift_history, avg_lift)
            
        if needs_std_moment and self._mcomp and self.moment_history:
            if not needs_avg_moment:  # Calculate avg_moment if not already done
                avg_moment = trapezoid(y=self.moment_history, x=self.force_times) / delta_t
            else:
                avg_moment = variables['avg_moment']
            variables['std_moment'] = self._compute_std(self.moment_history, avg_moment)
        elif needs_std_moment:
            variables['std_moment'] = 0.0
        
        # Normalize values if configured
        if self.normalize_reward:
            # Only normalize values that are used and have reference values
            for var_name, value in list(variables.items()):
                base_name = var_name.split('_')[1]  # Extract 'drag', 'lift', etc.
                if base_name in self.ref_values and self.ref_values[base_name] != 0:
                    variables[var_name] = value / self.ref_values[base_name]

        #print(f"manual reward={-np.abs(variables.get('avg_moment'))-2.0*variables.get('std_moment')}")
        #print(f"manual reward={-np.abs(variables.get('avg_moment')+0.01)-2.0*variables.get('std_moment')**2}")
        env = self.env
        if env and env.current_targets is not None:
            variables['tgt'] = list(env.current_targets)

        try:
            # Safely evaluate the reward function using AST
            reward = float(self.expr_evaluator.evaluate(self.reward_function, variables))
            #print(f"ast reward = {reward}")
            return reward
        except Exception as e:
            raise RuntimeError(
                f"Failed to evaluate reward function '{self.reward_function}': {e}"
            ) from e

    def reset(self):
        #self.latest_observation.zero_()
        self.force_times.clear()
        self.drag_history.clear()
        self.lift_history.clear()
        self.moment_history.clear()
        self.action_history.clear()
        self.sumabsact = 0.0
        
    def stress_tensor(self, u, du):
        c = self._constants

        # Density, energy
        rho, E = u[0], u[-1]

        # Gradient of density and momentum
        gradrho, gradrhou = du[:, 0], du[:, 1:-1]

        # Gradient of velocity
        gradu = (gradrhou - gradrho[:, None]*u[None, 1:-1]/rho) / rho

        # Bulk tensor
        bulk = np.eye(self.ndims)[:, :, None, None]*np.trace(gradu)

        # Viscosity
        mu = c['mu']

        if self._viscorr == 'sutherland':
            cpT = c['gamma']*(E/rho - 0.5*np.sum(u[1:-1]**2, axis=0)/rho**2)
            Trat = cpT/c['cpTref']
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        return -mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)

    def ac_stress_tensor(self, du):
        # Gradient of velocity and kinematic viscosity
        gradu, nu = du[:, 1:], self._constants['nu']

        return -nu*(gradu + gradu.swapaxes(0, 1))
    
class SafeExpressionEvaluator:
    """
    Evaluate a restricted mathematical expression safely.

    Allowed:
        • numeric literals
        • variables passed in `variables`
        • +  –  *  /  **  unary ±
        • abs, sqrt, exp, log, log10, sin, cos, tan, min, max
        • constants: pi, e
    Anything else (attributes, subscripts, comparisons, etc.) raises ValueError.
    """

    _OPS = {
        ast.Add:  operator.add,
        ast.Sub:  operator.sub,
        ast.Mult: operator.mul,
        ast.Div:  operator.truediv,
        ast.Pow:  operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    _FUNCS = {
        'abs':   abs,
        'sqrt':  math.sqrt,
        'exp':   math.exp,
        'log':   math.log,      # natural
        'log10': math.log10,
        'sin':   math.sin,
        'cos':   math.cos,
        'tan':   math.tan,
        'min':   min,
        'max':   max,
    }

    _CONST = {'pi': math.pi, 'e': math.e}

    # ------------------------------------------------------------------ public
    def evaluate(self, expr: str, variables: dict | None = None) -> float:
        """Return the numeric value of *expr* given *variables*."""
        variables = variables or {}

        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError as err:
            raise ValueError(f"Invalid expression: {err}") from None

        return float(self._eval(tree.body, variables))

    def find_used_variables(self, expr: str) -> set[str]:
        """Return the set of variable names that the expression contains."""
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            return set()

        vars_: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in self._FUNCS and node.id not in self._CONST:
                    vars_.add(node.id)
        return vars_

    # ----------------------------------------------------------------- private
    def _eval(self, node: ast.AST, env: dict[str, float]) -> float:
        """Recursive AST interpreter."""

        # ----- literals -------------------------------------------------------
        if isinstance(node, ast.Constant):        # Py ≥ 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric literals are allowed")

        # ----- identifiers ----------------------------------------------------
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self._CONST:
                return self._CONST[node.id]
            raise ValueError(f"Unknown variable: {node.id}")

        # ----- subscriptions like tgt[0] -------------------------------------
        if isinstance(node, ast.Subscript):
            base = self._eval(node.value, env)

            # Only allow constant integer indices
            idx_node = node.slice
            if isinstance(idx_node, ast.Constant):
                idx_value = idx_node.value
            elif hasattr(ast, 'Index') and isinstance(idx_node, ast.Index):
                # ast.Index exists on Python <3.9; keep for completeness
                inner = idx_node.value
                if isinstance(inner, ast.Constant):
                    idx_value = inner.value
                else:
                    raise ValueError("Only constant indices are supported in subscriptions")
            else:
                raise ValueError("Only constant indices are supported in subscriptions")

            if not isinstance(idx_value, int):
                raise ValueError("Subscript indices must be integers")

            try:
                return base[idx_value]
            except Exception as exc:
                raise ValueError(f"Invalid subscript access: {exc}") from None

        # ----- unary + / - ----------------------------------------------------
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.operand, env))

        # ----- binary operators ----------------------------------------------
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPS:
            left  = self._eval(node.left,  env)
            right = self._eval(node.right, env)
            return self._OPS[type(node.op)](left, right)

        # ----- function calls -------------------------------------------------
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname not in self._FUNCS:
                raise ValueError(f"Function not allowed: {fname}")
            args = [self._eval(arg, env) for arg in node.args]
            return self._FUNCS[fname](*args)

        # ----- anything else --------------------------------------------------
        raise ValueError(f"Unsupported expression element: {ast.dump(node, annotate_fields=False)}")
