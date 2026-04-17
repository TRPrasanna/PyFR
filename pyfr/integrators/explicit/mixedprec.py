import numpy as np

from pyfr.integrators.base import _common_plugin_prop


class _Fp64Kernel:
    """Lightweight kernel-like wrapper for a Python closure.

    The explicit RKVdH2R stepper composes fp64 stage updates by pushing a
    list of these objects through ``backend.run_kernels``.  Each instance
    stores the matrices it touches (so backends that track dependencies
    see it behave like a regular kernel), a ``bind`` hook for the runtime
    ``dt`` argument, and a ``run`` callable that performs the fp64 math.
    """

    def __init__(self, mats, run):
        self.mats = list(mats)
        self._run = run
        self._bound = {}

    def bind(self, **kwargs):
        self._bound.update(kwargs)

    def run(self):
        self._run(**self._bound)


class MixedPrecisionExplicitMixin:
    """Explicit integrator mixin that carries the stage/state through fp64.

    The backend remains at single precision so that all operator/RHS
    evaluations continue to run in fp32.  Alongside the fp32 register
    banks, we keep a parallel set of fp64 "mirror" arrays which are the
    authoritative copy of the state across stages and timesteps.

    Invariant (after every mutation):
        bank.data == mirror.astype(bank.dtype)

    Stage/step linear combinations are evaluated in fp64 on the mirrors
    and then the result is down-cast into the fp32 bank so that the
    existing fp32 RHS kernels keep seeing a consistent input.  After
    each RHS evaluation the fp32 bank is promoted back into the mirror.
    """

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        if not getattr(self.backend, 'mixed_precision', False):
            self._mp_active = False
            return

        if self.backend.name != 'openmp':
            raise NotImplementedError(
                "mixed-precision is currently only implemented for the "
                f"'openmp' backend, got {self.backend.name!r}"
            )

        # Allocate fp64 mirrors, one per bank, initialised from the
        # backend-precision bank contents.
        self._mp_mirrors = []
        for ebanks in self.system.ele_banks:
            self._mp_mirrors.append([
                np.array(b.data, dtype=np.float64, copy=True) for b in ebanks
            ])

        # On restart, the current register was seeded from the fp32 bank
        # (populated via an fp64→fp32 downcast in alloc_bank).  Re-install
        # the fp64 initial condition in the mirror so the preserved fp64
        # history survives the checkpoint/restart boundary — this is only
        # a real gain if the checkpoint was itself written in fp64 (which
        # the writer plugin does when mixed-precision is active).
        #
        # ``system.ele_map`` is deleted by ``system.commit()``, so the
        # Elements instances are gone by now.  In a matching-basis restart
        # ``initsoln.data[etype]`` is already the solution on the current
        # soln points with shape (nupts, nvars, neles); we only need to
        # reshape + pack it.  A polynomial-order change across a
        # mixed-precision restart is not supported and we bail explicitly.
        if initsoln is not None:
            for eidx, etype in enumerate(self.system.ele_types):
                data = initsoln.data.get(etype)
                if data is None:
                    continue

                bank = self.system.ele_banks[eidx][self.idxcurr]
                nupts, nvars, neles = bank.ioshape
                if data.shape != (nupts, nvars, neles):
                    raise NotImplementedError(
                        'mixed-precision restart requires matching solution '
                        f'basis (etype {etype}: expected '
                        f'{(nupts, nvars, neles)}, got {data.shape})'
                    )

                ic64 = np.ascontiguousarray(data, dtype=np.float64)

                # ``_pack`` returns an ``ascontiguousarray(..., dtype=self.dtype)``
                # on its default path, which silently downcasts to the fp32
                # backend precision.  Route through the ``out=`` branch so the
                # packing assignment preserves fp64.
                packed = np.empty_like(self._mp_mirrors[eidx][self.idxcurr])
                bank._pack(ic64, out=packed)
                self._mp_mirrors[eidx][self.idxcurr] = packed
                bank.data[:] = packed.astype(bank.dtype, copy=False)

        self._mp_active = True

    # ------------------------------------------------------------------
    # Integrator-level hooks (exposed to plugins / writer)
    # ------------------------------------------------------------------
    def _mp_unpacked_soln(self):
        """Return a list of (nupts, nvars, neles) fp64 arrays for idxcurr."""
        out = []
        for mirror_list, ebanks in zip(self._mp_mirrors, self.system.ele_banks):
            bank = ebanks[self.idxcurr]
            out.append(np.ascontiguousarray(
                bank._unpack(mirror_list[self.idxcurr])
            ))
        return out

    @_common_plugin_prop('_curr_soln')
    def soln(self):
        if not getattr(self, '_mp_active', False):
            self.system.postproc(self.idxcurr)
            return self.system.ele_scal_upts(self.idxcurr)

        self._mp_postproc_and_sync(self.idxcurr)
        return self._mp_unpacked_soln()

    @_common_plugin_prop('_curr_grad_soln')
    def grad_soln(self):
        if getattr(self, '_mp_active', False):
            self._mp_postproc_and_sync(self.idxcurr)
        else:
            self.system.postproc(self.idxcurr)
        self.compute_grads()
        return [e.get() for e in self.system.eles_vect_upts]

    def _mp_postproc_and_sync(self, regidx):
        # Snapshot the fp32 bank, run postproc, and only promote the bank
        # back into the fp64 mirror for banks that postproc actually
        # mutated.  An unconditional sync_up would round-trip unchanged
        # banks through fp32 and quantize the mirror to fp32 ulp.
        pre = [ebanks[regidx].data.copy()
               for ebanks in self.system.ele_banks]
        self.system.postproc(regidx)
        for pre_data, mirror_list, ebanks in zip(pre, self._mp_mirrors,
                                                 self.system.ele_banks):
            post_data = ebanks[regidx].data
            if not np.array_equal(pre_data, post_data):
                mirror_list[regidx][:] = post_data.astype(np.float64,
                                                          copy=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mp_sync_down(self, regidx):
        """Down-cast the fp64 mirror for ``regidx`` into the fp32 bank."""
        for mirror_list, ebanks in zip(self._mp_mirrors, self.system.ele_banks):
            ebanks[regidx].data[:] = mirror_list[regidx].astype(
                ebanks[regidx].dtype, copy=False
            )

    def _mp_sync_up(self, regidx):
        """Promote the fp32 bank for ``regidx`` into the fp64 mirror."""
        for mirror_list, ebanks in zip(self._mp_mirrors, self.system.ele_banks):
            mirror_list[regidx][:] = ebanks[regidx].data.astype(
                np.float64, copy=False
            )

    # ------------------------------------------------------------------
    # Overridden integrator primitives
    # ------------------------------------------------------------------
    def _rhs(self, t, uin, uout):
        if not getattr(self, '_mp_active', False):
            return super()._rhs(t, uin, uout)

        # The fp32 bank for uin already holds the fp32 down-cast of the
        # authoritative fp64 mirror (invariant maintained elsewhere).
        super()._rhs(t, uin, uout)

        # Promote the fp32 RHS output into the fp64 mirror so subsequent
        # stage updates accumulate in fp64.
        self._mp_sync_up(uout)

    def _addv(self, consts, regidxs, in_scale=(), in_scale_idxs=(),
              out_scale=()):
        if not getattr(self, '_mp_active', False):
            return super()._addv(consts, regidxs, in_scale, in_scale_idxs,
                                 out_scale)

        if in_scale or out_scale:
            raise NotImplementedError(
                'mixed-precision axnpby does not support in_scale/out_scale'
            )

        if len(regidxs) != len(set(regidxs)):
            raise ValueError('Duplicate register indices')

        rout = regidxs[0]
        in_regs = regidxs[1:]
        c_out = float(consts[0])
        c_in = [float(c) for c in consts[1:]]

        for mirror_list, ebanks in zip(self._mp_mirrors, self.system.ele_banks):
            acc = c_out * mirror_list[rout] if c_out != 0.0 else None
            for c, r in zip(c_in, in_regs):
                term = c * mirror_list[r]
                acc = term if acc is None else acc + term

            if acc is None:
                acc = np.zeros_like(mirror_list[rout])

            mirror_list[rout] = acc
            # Maintain the invariant: down-cast into the fp32 bank.
            ebanks[rout].data[:] = acc.astype(ebanks[rout].dtype, copy=False)

    # ------------------------------------------------------------------
    # RKVdH2R pointwise kernel replacement (used by rk34/rk45)
    # ------------------------------------------------------------------
    def _get_rkvdh2_kerns(self, stage, r1, r2, rold=None, rerr=None):
        if not getattr(self, '_mp_active', False):
            return super()._get_rkvdh2_kerns(stage, r1, r2, rold, rerr)

        stepper = self
        nstages = stepper._nstages
        a = stepper.a
        b = stepper.b
        e = stepper.e

        errest = rold is not None
        kerns = []

        for mirror_list, ebanks in zip(stepper._mp_mirrors,
                                       stepper.system.ele_banks):
            mirrors = mirror_list
            banks = ebanks
            ir1, ir2, irold, irerr = r1, r2, rold, rerr

            def make_run(mirrors, banks, ir1, ir2, irold, irerr, stage,
                         errest, a, b, e, nstages):
                def run(dt=None):
                    if dt is None:
                        raise RuntimeError('rkvdh2 kernel not bound')

                    m1 = mirrors[ir1]
                    m2 = mirrors[ir2]

                    if errest and stage == 0:
                        mirrors[irerr] = dt*e[stage]*m2
                        mirrors[irold] = m1.copy()
                        banks[irerr].data[:] = \
                            mirrors[irerr].astype(banks[irerr].dtype)
                        banks[irold].data[:] = \
                            mirrors[irold].astype(banks[irold].dtype)
                    elif errest:
                        mirrors[irerr] = mirrors[irerr] + dt*e[stage]*m2
                        banks[irerr].data[:] = \
                            mirrors[irerr].astype(banks[irerr].dtype)

                    if stage < nstages - 1:
                        new_r1 = m1 + dt*a[stage]*m2
                        new_r2 = m1 + dt*b[stage]*m2
                        mirrors[ir1] = new_r1
                        mirrors[ir2] = new_r2
                        banks[ir1].data[:] = new_r1.astype(banks[ir1].dtype)
                        banks[ir2].data[:] = new_r2.astype(banks[ir2].dtype)
                    else:
                        new_r1 = m1 + dt*b[stage]*m2
                        mirrors[ir1] = new_r1
                        banks[ir1].data[:] = new_r1.astype(banks[ir1].dtype)

                return run

            mats = [banks[ir1], banks[ir2]]
            if errest:
                mats.extend([banks[irold], banks[irerr]])

            kerns.append(_Fp64Kernel(
                mats,
                make_run(mirrors, banks, ir1, ir2, irold, irerr, stage,
                         errest, a, b, e, nstages),
            ))

        return kerns
