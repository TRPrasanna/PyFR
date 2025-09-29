<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

## Externs:
##   act_pack      : [A, D, F, t0]
##   t_act_interval: [Δt]
## Compile-time constants supplied from inters.py via self.c:
##   Amin, Amax, Dmin, Dmax, Fmin, Fmax, Deps, and the jet direction fields c[u|v|w]

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur'
             externs='ploc, t, act_pack, t_act_interval'>
    // Unpack params and timing
    fpdtype_t A  = act_pack[0][0];
    fpdtype_t D  = act_pack[0][1];
    fpdtype_t F  = act_pack[0][2];
    fpdtype_t t0 = act_pack[0][3];
    fpdtype_t T  = t_act_interval[0][0];

    // Clamp parameters against per-jet bounds
    A = max(${c['Amin']}, min(${c['Amax']}, A));
    D = max(${c['Dmin']}, min(${c['Dmax']}, D));
    // Physical guard for duty
    D = max((fpdtype_t)0.0, min((fpdtype_t)1.0, D));
    D = max(D, (fpdtype_t)${c['Deps']});
    D = min(D, (fpdtype_t)(1.0 - ${c['Deps']}));
    // Frequency non-negative and bounded
    F = max(${c['Fmin']}, min(${c['Fmax']}, F));

    // Adaptive-stepping guards
    fpdtype_t tau = t - t0;
    if (tau <= 0) tau = 0;           // handle early calls / FP jitter
    if (tau >= T) {                   // past interval end: hard-zero control
        ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
        ur[${i + 1}] = -ul[${i + 1}];
% endfor
        ur[${nvars - 1}] = ul[${nvars - 1}];
        return;
    }

    // Square pulse with fixed D and F inside the interval
    fpdtype_t cycles = F * tau;
    fpdtype_t frac   = cycles - floor(cycles);
    fpdtype_t g = (frac < D) ? (fpdtype_t)1.0 : (fpdtype_t)0.0;

    fpdtype_t control = A * g;

    // Apply like type5 adiabatic jet
    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = -ul[${i + 1}] + 2.0*ul[0]*control*(${c[v]});
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur'
             externs='ploc, t, act_pack, t_act_interval'>
    fpdtype_t A  = act_pack[0][0];
    fpdtype_t D  = act_pack[0][1];
    fpdtype_t F  = act_pack[0][2];
    fpdtype_t t0 = act_pack[0][3];
    fpdtype_t T  = t_act_interval[0][0];

    A = max(${c['Amin']}, min(${c['Amax']}, A));
    D = max(${c['Dmin']}, min(${c['Dmax']}, D));
    D = max((fpdtype_t)0.0, min((fpdtype_t)1.0, D));
    D = max(D, (fpdtype_t)${c['Deps']});
    D = min(D, (fpdtype_t)(1.0 - ${c['Deps']}));
    F = max(${c['Fmin']}, min(${c['Fmax']}, F));

    fpdtype_t tau = t - t0;
    if (tau <= 0) tau = 0;
    if (tau >= T) {
        ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
        ur[${i + 1}] = 0.0;
% endfor
        ur[${nvars - 1}] = ul[${nvars - 1}]
                         - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                         + (0.5/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
        return;
    }

    fpdtype_t cycles = F * tau;
    fpdtype_t frac   = cycles - floor(cycles);
    fpdtype_t g = (frac < D) ? (fpdtype_t)1.0 : (fpdtype_t)0.0;
    fpdtype_t control = A * g;

    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = control * ul[0] * (${c[v]});
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + (0.5/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_grad_state' params='ul, nl, grad_ul, grad_ur'>
    ${pyfr.expand('bc_common_grad_copy', 'ul', 'nl', 'grad_ul', 'grad_ur')};
</%pyfr:macro>
