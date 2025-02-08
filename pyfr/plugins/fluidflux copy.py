from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, SurfaceMixin, init_csv


class FluidFluxPlugin(SurfaceMixin, BaseSolnPlugin):
    name = 'fluidflux'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Check if the system is incompressible
        self._ac = intg.system.name.startswith('ac')

        # Viscous correction
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Boundary to integrate over
        bc = f'bcon_{suffix}_p{intg.rallocs.prank}'

        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # See which ranks have the boundary
        bcranks = comm.gather(bc in mesh, root=root)

        # The root rank needs to open the output file
        if rank == root:
            if not any(bcranks):
                raise RuntimeError(f'Boundary {suffix} does not exist')

            # CSV header
            header = ['t', 'massflux'][:2]

            # Open
            self.outf = init_csv(self.cfg, cfgsect, ','.join(header))

        # Interpolation matrices and quadrature weights
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)

        # If we have the boundary then process the interface
        if bc in mesh:
            # Element indices and associated face normals
            eidxs = defaultdict(list)
            norms = defaultdict(list)

            for etype, eidx, fidx, flags in mesh[bc].tolist():
                eles = elemap[etype]
                itype, proj, norm = eles.basis.faces[fidx]

                ppts, pwts = self._surf_quad(itype, proj, flags='s')
                nppts = len(ppts)

                # Get phyical normals
                pnorm = eles.pnorm_at(ppts, [norm]*nppts)[:, eidx]

                eidxs[etype, fidx].append(eidx)
                norms[etype, fidx].append(pnorm)

                if (etype, fidx) not in m0:
                    m0[etype, fidx] = eles.basis.ubasis.nodal_basis_at(ppts)
                    qwts[etype, fidx] = pwts

            self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
            self._norms = {k: np.array(v) for k, v in norms.items()}

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))
        nvars = self.nvars
        # Flux
        f = np.zeros(1)

        for etype, fidx in self._m0:
            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Convert conservative variables to primitive variables
            pri_vars = self.elementscls.con_to_pri(ufpts, self.cfg)
            vs = np.array(pri_vars[1:-1])  # Velocity components

            if self._ac:
                # Artificial compressibility: density is not a primitive variable
                # Set rho to 1 with the appropriate shape
                rho = np.ones_like(vs[0])
            else:
                # Compressible flow: extract rho from primitive variables
                rho = pri_vars[0]

            # Compute rhovs: multiply rho and vs
            rhovs = rho[None, :, :] * vs

            # Get the quadrature weights and normal vectors
            qwts = self._qwts[etype, fidx]
            norms = self._norms[etype, fidx]
            # Do the quadrature
            f[0] += np.einsum('i,jim,mij->', qwts, rhovs, norms)

        # Reduce and output if we're the root rank
        if rank != root:
            comm.Reduce(f, None, op=mpi.SUM, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, f, op=mpi.SUM, root=root)

            # Write
            print(intg.tcurr, *f.ravel(), sep=',', file=self.outf)

            # Flush to disk
            self.outf.flush()
