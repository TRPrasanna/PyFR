import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.common import DatasetAppender, init_csv, open_hdf5_a
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.plugins.soln.fluidforce import FluidForceIntegrator


class FluidFluxPlugin(BaseSolnPlugin):
    name = 'fluidflux'
    systems = 'euler|navier-stokes'
    formulations = ['dual', 'std']
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Output frequency
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Check if the system is incompressible
        self._ac = intg.system.name.startswith('ac')

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Moments
        mcomp = 3 if self.ndims == 3 else 1
        self._mcomp = mcomp if self.cfg.hasopt(cfgsect, 'morigin') else 0
        morigin = None
        if self._mcomp:
            morigin = np.array(self.cfg.getliteral(cfgsect, 'morigin'))
            if len(morigin) != self.ndims:
                raise ValueError(f'morigin must have {self.ndims} components')

        # See which ranks have the boundary
        bcranks = comm.gather(suffix in intg.system.mesh.bcon, root=root)

        # The root rank needs to open the output file
        if rank == root:
            if not any(bcranks):
                raise RuntimeError(f'Boundary {suffix} does not exist')

            match self.cfg.get(cfgsect, 'file-format', 'csv'):
                case 'csv':
                    self._init_csv()
                case 'hdf5':
                    self._init_hdf5()
                case _:
                    raise ValueError('Invalid file format')

        # Set interpolation matrices and quadrature weights
        self.ff_int = FluidForceIntegrator(self.cfg, cfgsect, intg.system,
                                           suffix, morigin)

    @property
    def _header(self):
        header = ['t', 'massflux']
        header += [f'momflux_{v}' for v in 'xyz'[:self.ndims]]

        if self._mcomp:
            if self.ndims == 2:
                header += ['moment_z']
            else:
                header += ['moment_x', 'moment_y', 'moment_z']

        return ','.join(header)

    def _init_csv(self):
        self.csv = init_csv(self.cfg, self.cfgsect, self._header, nflush=1)
        self._write = self._write_csv

    def _write_csv(self, t, flux):
        self.csv(t, *flux.ravel())

    def _init_hdf5(self):
        outf = open_hdf5_a(self.cfg.get(self.cfgsect, 'file'))
        # HDF5 rows store time plus the full flux vector.
        nvars = 2 + self.ndims + self._mcomp

        dset = self.cfg.get(self.cfgsect, 'file-dataset')
        if dset in outf:
            ff = outf[dset]

            if ff.shape[1] != nvars:
                raise ValueError('Invalid dataset')
        else:
            ff = outf.create_dataset(dset, (0, nvars), float,
                                     chunks=(128, nvars),
                                     maxshape=(None, nvars))
            ff.dims[1].label = self._header

        self._flux = DatasetAppender(ff)
        self._write = self._write_hdf5

    def _write_hdf5(self, t, flux):
        self._flux(np.concatenate(([t], flux.ravel())))

    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        ndims, nvars, mcomp = self.ndims, self.nvars, self._mcomp

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))

        # [mass flux, momentum flux (ndims), moments (mcomp)]
        flux = np.zeros(1 + ndims + mcomp)

        for (etype, fidx), m0 in self.ff_int.m0.items():
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self.ff_int.eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            # Convert conservative variables to primitive variables
            pri_vars = self.elementscls.con_to_pri(ufpts, self.cfg)
            vs = np.array(pri_vars[1:-1])

            # rho for compressible; rho = 1 for AC formulations
            if self._ac:
                rho = np.ones_like(vs[0])
            else:
                rho = pri_vars[0]

            rhovs = rho[None, :, :] * vs

            # Get the quadrature weights and normal vectors
            qwts = self.ff_int.qwts[etype, fidx]
            norms = self.ff_int.norms[etype, fidx]

            # 1) Mass flux: \int rho v \cdot n dS
            flux[0] += np.einsum('i,jim,mij->', qwts, rhovs, norms)

            # 2) Momentum flux: \int rho v_k (v \cdot n) dS
            flux[1:ndims + 1] += np.einsum('i,jim,mij,kim->k',
                                           qwts, rhovs, norms, vs)

            if self._mcomp:
                # Flux points positions relative to the moment origin
                rfpts = self.ff_int.rfpts[etype, fidx]

                # Normal momentum flux vector at each flux point
                momflux = np.einsum('jim,mij,kim->kim', rhovs, norms, vs)

                # Moments from r x (momentum flux)
                rcf = np.atleast_3d(np.cross(rfpts, momflux.T))
                flux[ndims + 1:] += np.einsum('i,jik->k', qwts, rcf)

        # Reduce and output if we're the root rank
        if rank != root:
            comm.Reduce(flux, None, op=mpi.SUM, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, flux, op=mpi.SUM, root=root)
            self._write(intg.tcurr, flux)
