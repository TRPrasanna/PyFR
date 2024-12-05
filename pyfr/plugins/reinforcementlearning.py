from pyfr.plugins.base import BaseSolverPlugin
from pyfr.quadrules import get_quadrule
from pyfr.plugins.sampler import _closest_pts, _plocs_to_tlocs
from pyfr.mpiutil import get_comm_rank_root, mpi
import numpy as np
import torch
from collections import defaultdict
from pyfr.plugins.base import BaseSolnPlugin, SurfaceMixin

class ReinforcementLearningPlugin(BaseSolverPlugin, SurfaceMixin, BaseSolnPlugin):
    name = 'reinforcementlearning'
    systems = ['ac-navier-stokes', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [2, 3]
    
    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get sampling points configuration
        self.pts = self.cfg.getliteral(cfgsect, 'probe-pts')
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')
        
        # Setup sampling infrastructure
        self._setup_sampling(intg)
        
        # Calculate observation size based on probe points and variables
        nvars = len(self.elementscls.privarmap[self.ndims]) if self.fmt == 'primitive' else len(self.elementscls.convarmap[self.ndims])
        self.observation_size = len(self.pts) * nvars
        self.nvars = nvars
        
        # Rest of initialization
        self.action_interval = self.cfg.getfloat(cfgsect, 'action-interval', 0.1)
        self.control_signal = torch.tensor([0.0], device=self.device)
        self.last_action_time = intg.tcurr
        self.latest_observation = torch.zeros(self.observation_size, device=self.device)
        self.latest_reward = 0.0

        # Force calculation setup (from FluidForcePlugin)
        self._viscous = 'navier-stokes' in intg.system.name
        self._ac = intg.system.name.startswith('ac')
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')
        self._constants = self.cfg.items_as('constants', float)

        self.surf_bname = self.cfg.get(cfgsect, 'surface', None)
        if not self.surf_bname:
            raise ValueError("No surface specified for drag calculation")
            
        print(f"Calculating drag over boundary: {self.surf_bname}")
        # Get boundary info with specified name
        bc = f'bcon_{self.surf_bname}_p{intg.rallocs.prank}'
        if bc not in intg.system.mesh:
            raise ValueError(f"Boundary '{self.surf_bname}' not found")
        mesh, elemap = intg.system.mesh, intg.system.ele_map
        
        # Initialize force calculation matrices
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)
        if self._viscous:
            self._m4 = m4 = {}
            rcpjact = {}
            
        # Process boundary (similar to FluidForcePlugin)
        if bc in mesh:
            eidxs = defaultdict(list)
            norms = defaultdict(list)
            
            for etype, eidx, fidx, flags in mesh[bc].tolist():
                eles = elemap[etype]
                itype, proj, norm = eles.basis.faces[fidx]
                
                # Get quadrature points
                ppts, pwts = self._surf_quad(itype, proj, flags='s')
                nppts = len(ppts)
                
                # Get physical normals
                pnorm = eles.pnorm_at(ppts, [norm]*nppts)[:, eidx]
                
                eidxs[etype, fidx].append(eidx)
                norms[etype, fidx].append(pnorm)
                
                if (etype, fidx) not in m0:
                    m0[etype, fidx] = eles.basis.ubasis.nodal_basis_at(ppts)
                    qwts[etype, fidx] = pwts
                    
                if self._viscous and etype not in m4:
                    m4[etype] = eles.basis.m4
                    smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
                    rcpdjac = eles.rcpdjac_at_np('upts')
                    rcpjact[etype] = smat*rcpdjac
                    
            self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
            self._norms = {k: np.array(v) for k, v in norms.items()}
            if self._viscous:
                self._rcpjact = {k: rcpjact[k[0]][..., v] 
                                for k, v in self._eidxs.items()}

    def __call__(self, intg):
        # Update observations and rewards at action times
        if intg.tcurr >= self.last_action_time + self.action_interval:
            self.latest_observation = self._get_observation(intg)
            self.latest_reward = self._get_reward(intg)
            self.last_action_time = intg.tcurr

    def set_action(self, action):
        self.control_signal = torch.tensor([action], device=self.device)

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
            
        # Convert to tensor
        obs = torch.tensor(samples, device=self.device).flatten()
        return obs

    def _get_reward(self, solver):
        """Compute reward as drag force"""
        comm, rank, root = get_comm_rank_root()
        
        # Get solution matrices
        solns = dict(zip(solver.system.ele_types, solver.soln))
        ndims = self.ndims
        
        # Force vectors (pressure and viscous)
        fm = np.zeros((2 if self._viscous else 1, ndims))
        
        for etype, fidx in self._m0:
            # Get interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape
            
            # Get solution at points
            uupts = solns[etype][..., self._eidxs[etype, fidx]]
            
            # Interpolate to face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, self.nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)
            
            # Compute pressure
            pidx = 0 if self._ac else -1
            p = self.elementscls.con_to_pri(ufpts, self.cfg)[pidx]
            
            # Get weights and normals
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]
            
            # Pressure force
            fm[0, :ndims] += np.einsum('i...,ij,jik', qwts, p, norms)
            
            if self._viscous:
                # Get operator and J^-T matrix
                m4 = self._m4[etype]
                rcpjact = self._rcpjact[etype, fidx]

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

                # Do the quadrature
                fm[1, :ndims] += np.einsum('i...,klij,jil', qwts, vis, norms)
                
        # Reduce across ranks
        if rank != root:
            comm.Reduce(fm, None, op=mpi.SUM, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, fm, op=mpi.SUM, root=root)
            
        # Return negative drag (x-component)
        drag = -(fm[0, 0] + (fm[1, 0] if self._viscous else 0))
        return float(drag)
        

    def reset(self):
        self.control_signal = torch.tensor([0.0], device=self.device)
        self.latest_observation.zero_()
        self.latest_reward = 0.0
        self.last_action_time = 0.0  # Reset time tracking

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
