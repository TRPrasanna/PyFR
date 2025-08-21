## pyfr/plugins/kernels/neuralsource.mako
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name="neuralsource"
             params="t, u, ploc, src"
             externs="control_params_src, t_act_interval_src">

    // Three linear ramps with clamping for channels 0..2
    fpdtype_t c0 = (control_params_src[0][1] - control_params_src[0][0]) / t_act_interval_src[0][0]
                   * (t - control_params_src[0][2]) + control_params_src[0][0];
    fpdtype_t lo0 = min(control_params_src[0][0], control_params_src[0][1]);
    fpdtype_t hi0 = max(control_params_src[0][0], control_params_src[0][1]);
    c0 = max(lo0, min(c0, hi0));

    fpdtype_t c1 = (control_params_src[1][1] - control_params_src[1][0]) / t_act_interval_src[0][0]
                   * (t - control_params_src[1][2]) + control_params_src[1][0];
    fpdtype_t lo1 = min(control_params_src[1][0], control_params_src[1][1]);
    fpdtype_t hi1 = max(control_params_src[1][0], control_params_src[1][1]);
    c1 = max(lo1, min(c1, hi1));

    fpdtype_t c2 = (control_params_src[2][1] - control_params_src[2][0]) / t_act_interval_src[0][0]
                   * (t - control_params_src[2][2]) + control_params_src[2][0];
    fpdtype_t lo2 = min(control_params_src[2][0], control_params_src[2][1]);
    fpdtype_t hi2 = max(control_params_src[2][0], control_params_src[2][1]);
    c2 = max(lo2, min(c2, hi2));

    // User-provided masks for three segments
    fpdtype_t mask1 = ${mask1};
    fpdtype_t mask2 = ${mask2};
    fpdtype_t mask3 = ${mask3};

    // We drive the Y-direction force only
    fpdtype_t Sx = (fpdtype_t)0.0;
    fpdtype_t Sy = c0 * mask1 + c1 * mask2 + c2 * mask3;

    // Conservative vars and work term
    fpdtype_t rho  = u[0];
    fpdtype_t rhou = u[1];
    fpdtype_t rhov = u[2];
    fpdtype_t uvel = rhou / rho;
    fpdtype_t vvel = rhov / rho;

    // Add to RHS (no mass source)
    src[0] += (fpdtype_t)0.0;               // rho
    src[1] += Sx;                           // rhou (zero)
    src[2] += Sy;                           // rhov
% if ndims == 3:
    src[3] += (fpdtype_t)0.0;               // rhow
    src[${nvars - 1}] += uvel * Sx + vvel * Sy;  // E
% else:
    src[${nvars - 1}] += uvel * Sx + vvel * Sy;  // E
% endif
</%pyfr:macro>
