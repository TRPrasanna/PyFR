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


class NavierStokesSubInflowFrvNeuralBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv-neural'  
    cflux_state = 'ghost'
    def __init__(self, intg, be, lhs, elemap, cfgsect, cfg):
        self.backend = be
        super().__init__(be, lhs, elemap, cfgsect, cfg)
        self.intg = intg
        
        # Basic initialization
        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )

        # Neural network parameters
        self.nn_params = self.backend.matrix((1, 2), tags={'align'})
        self._set_external('nn_params', 'broadcast fpdtype_t[1][2]', value=self.nn_params)

        # Get sampling config
        self.pts = cfg.getliteral(cfgsect, 'samp-pts')
        self.samp_dt = cfg.getfloat(cfgsect, 'samp-dt', 0.1)
        
        # Initialize last sample time to current simulation time
        self._last_sample_time = intg.tcurr
            
        # Initialize sampling points
        self._initialize_sampling(elemap)

    def _initialize_sampling(self, elemap):
        """Initialize sampling with provided element map"""
        # MPI setup
        comm, rank, root = get_comm_rank_root()
        
        # Initialize sampling points
        elepts = [[] for i in range(len(elemap))]
        
        # Search locations using elemap directly
        tlocs, plocs = self._search_pts_init(elemap)
        closest = self._closest_pts(plocs, self.pts)
        
        # Process points
        for i, (dist, etype, (uidx, eidx)) in enumerate(closest):
            _, mrank = comm.allreduce((dist, rank), op=mpi.MINLOC)
            if rank == mrank:
                elepts[etype].append((i, eidx, tlocs[etype][uidx]))

        # Store refined points
        self._ourpts = self._refine_pts_init(elemap, elepts)

    def _search_pts_init(self, elemap):
        """Get search locations using element map"""
        tlocs, plocs = [], []

        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }

        for etype, eles in elemap.items():
            pts = get_quadrule(etype, qrule_map[etype], eles.basis.nupts).pts
            tlocs.append(pts)
            plocs.append(eles.ploc_at_np(pts).swapaxes(1, 2))

        return tlocs, plocs

    def _refine_pts_init(self, elemap, elepts):
        """Refine points using provided element map"""
        elelist = elemap.values()
        ptsinfo = []

        for etype, (eles, epts) in enumerate(zip(elelist, elepts)):
            if not epts:
                continue

            idx, eidx, tlocs = zip(*epts)
            spts = eles.eles[:, eidx, :]
            plocs = [self.pts[i] for i in idx]

            ntlocs, nplocs = self._plocs_to_tlocs(eles.basis.sbasis, spts, plocs, tlocs)
            intops = eles.basis.ubasis.nodal_basis_at(ntlocs)
            
            ptsinfo.extend((*info, etype) for info in zip(idx, eidx, nplocs, intops))

        ptsinfo.sort()
        return [(etype, *info) for idx, *info, etype in ptsinfo]

    def _closest_pts(self, epts, pts):
        ndims = len(pts[0])
        props = Property(dimension=ndims, interleaved=True)
        trees = []

        for e in epts:
            # Build list of solution points for each element type
            ins = [(i, [*p, *p], None) for i, p in enumerate(e.reshape(-1, ndims))]

            # Build tree of solution points for each element type
            trees.append(Index(ins, properties=props))

        for p in pts:
            # Find index of solution point closest to p
            amins = [np.unravel_index(next(t.nearest([*p, *p], 1)), ept.shape[:2])
                    for ept, t in zip(epts, trees)]

            # Find distance of solution point closest to p
            dmins = [np.linalg.norm(e[a] - p) for e, a in zip(epts, amins)]

            # Reduce across element types
            yield min(zip(dmins, range(len(epts)), amins))


    def _plocs_to_tlocs(self, sbasis, spts, plocs, tlocs):
        plocs, itlocs = np.array(plocs), np.array(tlocs)

        # Evaluate the initial guesses
        iplocs = np.einsum('ij,jik->ik', sbasis.nodal_basis_at(itlocs), spts)

        # Iterates
        kplocs, ktlocs = iplocs.copy(), itlocs.copy()

        # Apply three iterations of Newton's method
        for k in range(3):
            jac_ops = sbasis.jac_nodal_basis_at(ktlocs)

            A = np.einsum('ijk,jkl->kli', jac_ops, spts)
            b = kplocs - plocs
            ktlocs -= np.linalg.solve(A, b[..., None]).squeeze()

            ops = sbasis.nodal_basis_at(ktlocs)
            np.einsum('ij,jik->ik', ops, spts, out=kplocs)

        # Compute the initial and final distances from the target location
        idists = np.linalg.norm(plocs - iplocs, axis=1)
        kdists = np.linalg.norm(plocs - kplocs, axis=1)

        # Replace any points which failed to converge with their initial guesses
        closer = np.where(idists < kdists)
        ktlocs[closer] = itlocs[closer]
        kplocs[closer] = iplocs[closer]

        return ktlocs, kplocs
    
    def prepare(self, t):
        """Called each timestep"""
        # Only sample if enough time has elapsed
        if t - self._last_sample_time >= self.samp_dt:
            # Get solution values at sample points
            solns = self.intg.soln
            samples = [op @ solns[et][:, :, ei] for et, ei, _, op in self._ourpts]
            
            # Convert to primitive variables
            samples = np.array(samples)
            if samples.size:
                samples = self.intg.system.elementscls.con_to_pri(samples.T, self.cfg)
                samples = np.array(samples).T

            # Print sampled values
            comm, rank, root = get_comm_rank_root()
            if rank == root:
                print(f"t={t:.3f}, Sampled values:", samples)

            # Update last sample time
            self._last_sample_time = t

            # Update neural network parameters based on observations
            nn_params = self.nn_params.get()
            nn_params[0][0] = self._compute_control(samples)
            self.nn_params.set(nn_params)

    def _compute_control(self, observations):
        """Compute control action based on observations"""
        # TODO: Implement RL-based control
        return 0.09  # Default value for now

