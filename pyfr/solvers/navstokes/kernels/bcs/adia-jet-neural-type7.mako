<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

## Externs:
##   act_pack      : [D, F, t0]
##   t_act_interval: [Δt]
## Compile-time constants supplied from inters.py via self.c:
##   Dmin, Dmax, Fmin, Fmax, Deps, and the slot profile fields c[u|v|w]

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur'
             externs='ploc, t, act_pack, t_act_interval'>
    // Unpack params and timing
    fpdtype_t D  = act_pack[0][0];
    fpdtype_t F  = act_pack[0][1];
    fpdtype_t t0 = act_pack[0][2];
    fpdtype_t T  = t_act_interval[0][0];

    // Clamp parameters against per-jet bounds
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
    if (tau >= T) {                  // past interval end: hard-zero control
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

    // Here amplitude is embedded in c[u|v|w] fields from cfg
    fpdtype_t control = g;

    // Apply like type5/6 adiabatic jet using slot profile fields
    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = -ul[${i + 1}] + 2.0*ul[0]*control*(${c[v]});
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur'
             externs='ploc, t, act_pack, t_act_interval'>
    fpdtype_t D  = act_pack[0][0];
    fpdtype_t F  = act_pack[0][1];
    fpdtype_t t0 = act_pack[0][2];
    fpdtype_t T  = t_act_interval[0][0];

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

    // Amplitude is embedded in c[u|v|w]
    fpdtype_t control = g;

    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = control * ul[0] * (${c[v]});
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + (0.5/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_grad_state' params='ul, nl, grad_ul, grad_ur'>
    fpdtype_t rcprho = 1.0/ul[0];

% if ndims == 2:
    fpdtype_t u = rcprho*ul[1], v = rcprho*ul[2];

    fpdtype_t u_x = grad_ul[0][1] - u*grad_ul[0][0];
    fpdtype_t u_y = grad_ul[1][1] - u*grad_ul[1][0];
    fpdtype_t v_x = grad_ul[0][2] - v*grad_ul[0][0];
    fpdtype_t v_y = grad_ul[1][2] - v*grad_ul[1][0];

    // Compute temperature derivatives (c_v*rho*dT/d[x,y,z])
    fpdtype_t Tl_x = grad_ul[0][3] - (rcprho*grad_ul[0][0]*ul[3]
                                      + u*u_x + v*v_x);
    fpdtype_t Tl_y = grad_ul[1][3] - (rcprho*grad_ul[1][0]*ul[3]
                                      + u*u_y + v*v_y);

    // Copy all fluid-side gradients across to wall-side gradients
    ${pyfr.expand('bc_common_grad_copy', 'ul', 'nl', 'grad_ul', 'grad_ur')};

    // Correct copied across in-fluid temp gradients to in-wall gradients
    grad_ur[0][3] -= nl[0]*nl[0]*Tl_x + nl[0]*nl[1]*Tl_y;
    grad_ur[1][3] -= nl[1]*nl[0]*Tl_x + nl[1]*nl[1]*Tl_y;

% elif ndims == 3:
    fpdtype_t u = rcprho*ul[1], v = rcprho*ul[2], w = rcprho*ul[3];

    // Velocity derivatives (rho*grad[u,v,w])
    fpdtype_t u_x = grad_ul[0][1] - u*grad_ul[0][0];
    fpdtype_t u_y = grad_ul[1][1] - u*grad_ul[1][0];
    fpdtype_t u_z = grad_ul[2][1] - u*grad_ul[2][0];
    fpdtype_t v_x = grad_ul[0][2] - v*grad_ul[0][0];
    fpdtype_t v_y = grad_ul[1][2] - v*grad_ul[1][0];
    fpdtype_t v_z = grad_ul[2][2] - v*grad_ul[2][0];
    fpdtype_t w_x = grad_ul[0][3] - w*grad_ul[0][0];
    fpdtype_t w_y = grad_ul[1][3] - w*grad_ul[1][0];
    fpdtype_t w_z = grad_ul[2][3] - w*grad_ul[2][0];

    // Compute temperature derivatives (c_v*rho*dT/d[x,y,z])
    fpdtype_t Tl_x = grad_ul[0][4] - (rcprho*grad_ul[0][0]*ul[4]
                                      + u*u_x + v*v_x + w*w_x);
    fpdtype_t Tl_y = grad_ul[1][4] - (rcprho*grad_ul[1][0]*ul[4]
                                      + u*u_y + v*v_y + w*w_y);
    fpdtype_t Tl_z = grad_ul[2][4] - (rcprho*grad_ul[2][0]*ul[4]
                                      + u*u_z + v*v_z + w*w_z);

    // Copy all fluid-side gradients across to wall-side gradients
    ${pyfr.expand('bc_common_grad_copy', 'ul', 'nl', 'grad_ul', 'grad_ur')};

    // Correct copied across in-fluid temp gradients to in-wall gradients
    grad_ur[0][4] -= nl[0]*nl[0]*Tl_x + nl[0]*nl[1]*Tl_y + nl[0]*nl[2]*Tl_z;
    grad_ur[1][4] -= nl[1]*nl[0]*Tl_x + nl[1]*nl[1]*Tl_y + nl[1]*nl[2]*Tl_z;
    grad_ur[2][4] -= nl[2]*nl[0]*Tl_x + nl[2]*nl[1]*Tl_y + nl[2]*nl[2]*Tl_z;
% endif
</%pyfr:macro>