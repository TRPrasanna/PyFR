## pyfr/plugins/kernels/neuralsource.mako
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name="neuralsource"
             params="t, u, ploc, src"
             externs="control_params_src, t_act_interval_src">

    // Linear ramps with clamping for the two control channels
    fpdtype_t c1 = (control_params_src[0][1] - control_params_src[0][0]) / t_act_interval_src[0][0]
                   * (t - control_params_src[0][2]) + control_params_src[0][0];
    fpdtype_t lo1 = min(control_params_src[0][0], control_params_src[0][1]);
    fpdtype_t hi1 = max(control_params_src[0][0], control_params_src[0][1]);
    c1 = max(lo1, min(c1, hi1));

    fpdtype_t c2 = (control_params_src[1][1] - control_params_src[1][0]) / t_act_interval_src[0][0]
                   * (t - control_params_src[1][2]) + control_params_src[1][0];
    fpdtype_t lo2 = min(control_params_src[1][0], control_params_src[1][1]);
    fpdtype_t hi2 = max(control_params_src[1][0], control_params_src[1][1]);
    c2 = max(lo2, min(c2, hi2));

    // User-provided masks
    fpdtype_t mask1 = ${mask1};
    fpdtype_t mask2 = ${mask2};

    fpdtype_t Sx = c1 * mask1 + c2 * mask2;

% if have_sy:
    fpdtype_t mask1y = ${mask1y};
    fpdtype_t mask2y = ${mask2y};
    fpdtype_t Sy = c1 * mask1y + c2 * mask2y;
% else:
    fpdtype_t Sy = (fpdtype_t)0.0;
% endif

    // Conservative vars and work term
    fpdtype_t rho  = u[0];
    fpdtype_t rhou = u[1];
    fpdtype_t rhov = u[2];
    fpdtype_t uvel = rhou / rho;
    fpdtype_t vvel = rhov / rho;

    // Add to RHS
    src[0] += (fpdtype_t)0.0;               // rho
    src[1] += Sx;                           // rhou
    src[2] += Sy;                           // rhov
% if ndims == 3:
    src[3] += (fpdtype_t)0.0;               // rhow (unused here)
    src[${nvars - 1}] += uvel * Sx + vvel * Sy;  // E
% else:
    src[${nvars - 1}] += uvel * Sx + vvel * Sy;  // E
% endif
</%pyfr:macro>
