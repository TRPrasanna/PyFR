<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t, control_params, t_act_interval'>
    fpdtype_t angl = ((control_params[0][1]-control_params[0][0])/t_act_interval[0][0]*(t-control_params[0][2]) + control_params[0][0])*0.0174532925199433;
    fpdtype_t lower_bound = min(control_params[0][0], control_params[0][1]);
    fpdtype_t upper_bound = max(control_params[0][0], control_params[0][1]);
    angl = max(lower_bound, min(control, upper_bound));
    ur[0] = ${c['rho']};
% if ndims == 2:
    ur[1] = cos(angl) * (${c['rho']}) * (${c['u']});
    ur[2] = sin(angl) * (${c['rho']}) * (${c['v']});
% elif ndims == 3:
    ur[1] = cos(angl) * (${c['rho']}) * (${c['u']});
    ur[2] = sin(angl) * (${c['rho']}) * (${c['v']});
    ur[3] = (${c['rho']}) * (${c['w']});
% endif
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>
