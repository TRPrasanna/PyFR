<% import numpy as np %> 

<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t, nn_params, jcenter, polarity'>
    <% angle_expr = "jcenter[0][0] + polarity[0][0]*nn_params[0][0]*0.0174532925199433" %>
    ur[0] = ${c['rho']};
% if ndims == 2:
    ur[1] = cos(${angle_expr}) * (${c['rho']}) * (${c['u']});
    ur[2] = sin(${angle_expr}) * (${c['rho']}) * (${c['v']});
% endif
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>
